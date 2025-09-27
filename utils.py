# Core libraries (from PDF.ipynb) + optimization libraries
import pymupdf4llm
import pathlib
import sys
import re
import os
import json
import time
import fitz 
from datetime import date, datetime
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Feature calculation libraries
import nltk
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict, stopwords
from lexicalrichness import LexicalRichness

# Database
from sqlalchemy import create_engine, Column, String, Integer, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError

# Machine Learning and Model Loading
import pickle
import joblib
import numpy as np

# OPTIMIZATION IMPORTS - for parallel processing within documents
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache, wraps
import hashlib


# Load system prompts
with open("instructions/text_extraction.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

with open("instructions/date_extraction.txt", "r", encoding="utf-8") as file:
    date_instruction = file.read()


def extract_pdf(input_file):
    pdf = pymupdf4llm.to_markdown(input_file)
    pattern = r'(\n\n-----\n\n)'
    parts = re.split(pattern, pdf)
    pages = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)]
    return pages


# Thread-safe cache for API responses
_api_cache = {}
_cache_lock = threading.Lock()

def _cache_key(text, prompt_type):
    """Generate cache key for API responses"""
    return hashlib.md5(f"{prompt_type}:{text[:500]}".encode()).hexdigest()

def clean_text(text):
    """EXACT same logic as PDF.ipynb with optional caching"""
    cache_key = _cache_key(text, "clean")
    
    with _cache_lock:
        if cache_key in _api_cache:
            return _api_cache[cache_key]
    
    completion = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", "content": text
            }
        ]
    )
    result = completion.choices[0].message.content
    
    with _cache_lock:
        _api_cache[cache_key] = result
    
    return result

def gpt_date_extract(text):
    """EXACT same logic as PDF.ipynb with optional caching"""
    cache_key = _cache_key(text, "date")
    
    with _cache_lock:
        if cache_key in _api_cache:
            return _api_cache[cache_key]
    
    completion = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": date_instruction
            },
            {
                "role": "user", "content": text
            }
        ]
    )
    result = completion.choices[0].message.content
    
    with _cache_lock:
        _api_cache[cache_key] = result
    
    return result


# ENHANCED version of parse_paper_content from PDF.ipynb
# Only modification: adds reference count parsing (minimal integration)
def parse_paper_content(text):
    # Regex pattern to extract elements between # and -----
    pattern = r'#.*?-----'
    
    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)
    
    dates = None
    main_content = None
    reference_count = None
    
    # Identify and assign the sections
    for match in matches:
        if match.startswith('#DATES:'):
            dates = match
        elif match.startswith('#MAIN_CONTENT:'):
            main_content = match
        elif match.startswith('#REFERENCE_COUNT:'):  # ONLY NEW ADDITION
            reference_count = match

    # Initialize the dictionary with default structure
    page_dict = {
        'DATES': {'Received': None, 'Accepted': None},
        'MAIN_CONTENT': None,
        'REFERENCE_COUNT': 0
    }
    
    # ====== DATES section ====== (EXACT same logic as PDF.ipynb)
    if dates:
        section_name = re.search(r'#(.*?):', dates).group(1).strip()
        pattern = r'-\s*(.*?):\s*(.*)'
        date_matches = re.findall(pattern, dates)
        page_dict[section_name] = {}  # Reset the section
        for key, value in date_matches:
            none_match = re.search(rf'\n-\s*{re.escape(key)}:\s*None', dates)
            page_dict[section_name][key.strip()] = None if none_match else value.strip()

    # ====== REFERENCE_COUNT section ====== (FROM features.ipynb)
    if reference_count:
        section_name = re.search(r'#(.*?):', reference_count).group(1).strip()
        # Match a single number after the dash
        match = re.search(r'-\s*(\d+)', reference_count)
        final_value = int(match.group(1)) if match else 0
        page_dict[section_name] = final_value

    # ====== MAIN_CONTENT section ====== (EXACT same logic as PDF.ipynb)
    if main_content:
        section_name = re.search(r'#(.*?):', main_content).group(1).strip()
        content = re.sub(r'#.*?:\s*', '', main_content)
        content = re.sub(r'-----$', '', content).strip()
        flattened_content = ' '.join(re.sub(r'-\s*', '', content).splitlines()).strip()
        flattened_content = re.sub(r'_+', '', flattened_content)
        flattened_content = re.sub(r'\*', '', flattened_content)
        flattened_content = re.sub(r'\s+', ' ', flattened_content)
        page_dict[section_name] = None if not flattened_content or flattened_content.lower() == "none" else flattened_content

    return page_dict


# EXACT COPY from PDF.ipynb - Critical date fallback function for accuracy
def check_date(filedict, input_file):
    received = False
    accepted = False

    for page_num in range(1, len(filedict) + 1):
        page_key = 'page' + str(page_num)
        if page_key in filedict and 'DATES' in filedict[page_key]:
            if filedict[page_key]['DATES'].get('Received') is not None:
                received = True
            if filedict[page_key]['DATES'].get('Accepted') is not None:
                accepted = True

    if received and accepted:
        return
    else:
        doc = fitz.open(input_file)

        # Create an array (list) to store the text of each page
        pages_text = []
        
        # Extract text from each page and store it in the list
        for page in doc:
            pages_text.append(page.get_text())

        for page_num, file in enumerate(pages_text, start=1):
            page_dict = {}
            cleaned_page = gpt_date_extract(file)
            
            if not cleaned_page or '#DATES:' not in cleaned_page:
                continue
                
            section_name = re.search(r'#(.*?):', cleaned_page).group(1).strip()
    
            # Extract the key-value pairs
            pattern = r'-\s*(.*?):\s*(.*)'
            matches = re.findall(pattern, cleaned_page)
            
            # Build the dictionary
            for key, value in matches:
                # Check if the value is explicitly marked as "None" (case-sensitive)
                none_match = re.search(rf'\n-\s*{re.escape(key)}:\s*None', cleaned_page)
                page_dict.setdefault(section_name, {})[key.strip()] = None if none_match else value.strip()

            if page_dict.get('DATES', {}).get('Received') is not None and page_dict.get('DATES', {}).get('Accepted') is not None:
                # Find the first page with DATES section to update
                for i in range(1, len(filedict) + 1):
                    page_key = 'page' + str(i)
                    if page_key in filedict and 'DATES' in filedict[page_key]:
                        filedict[page_key]['DATES']['Received'] = page_dict['DATES']['Received']
                        filedict[page_key]['DATES']['Accepted'] = page_dict['DATES']['Accepted']
                        doc.close()
                        return
        
        doc.close()


# OPTIMIZED version of process_pdf from PDF.ipynb
# PRESERVES EXACT SAME LOGIC with parallel page processing for speed

def process_single_page(page_data):
    """Process a single page - used for parallel processing"""
    page_num, page_content = page_data
    try:
        cleaned_page = clean_text(page_content)
        parsed_page = parse_paper_content(cleaned_page)
        return page_num, parsed_page
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        return page_num, {'DATES': {'Received': None, 'Accepted': None}, 'MAIN_CONTENT': None, 'REFERENCE_COUNT': 0}

def process_pdf_optimized(input_file, max_workers=4):
    """
    OPTIMIZED version of process_pdf from PDF.ipynb.
    Uses parallel processing for pages while preserving EXACT same logic.
    """
    file_dict = {}
    pdf = extract_pdf(input_file)
    
    # Prepare page data for parallel processing
    page_data = [(page_num, file_content) for page_num, file_content in enumerate(pdf, start=1)]
    
    # Process pages in parallel (significant speedup for multi-page docs)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {executor.submit(process_single_page, data): data[0] for data in page_data}
            
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    processed_page_num, parsed_page = future.result()
                    file_dict[f'page{processed_page_num}'] = parsed_page
                except Exception as e:
                    print(f"Error in parallel processing for page {page_num}: {e}")
                    # Fallback to sequential processing for this page
                    cleaned_page = clean_text(pdf[page_num-1])
                    parsed_page = parse_paper_content(cleaned_page)
                    file_dict[f'page{page_num}'] = parsed_page
    
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {e}")
        # Fallback to original sequential processing
        for page_num, file_content in enumerate(pdf, start=1):
            cleaned_page = clean_text(file_content)
            parsed_page = parse_paper_content(cleaned_page)
            file_dict[f'page{page_num}'] = parsed_page
    
    # EXACT same date checking logic as PDF.ipynb (critical for accuracy)
    check_date(file_dict, input_file)

    return file_dict

# Keep original function as fallback
def process_pdf_enhanced(input_file):
    """Original sequential version - kept as fallback"""
    return process_pdf_optimized(input_file, max_workers=1)


# ENHANCED version of consolidate from PDF.ipynb
# Only modification: adds reference count consolidation logic
def consolidate_enhanced(filedict):
    consolidated = {}

    # ====== Consolidate DATES ====== (EXACT same logic as PDF.ipynb with safety checks)
    for i in range(1, len(filedict) + 1):
        page_key = 'page' + str(i)
        if (page_key in filedict and 
            'DATES' in filedict[page_key] and 
            filedict[page_key]['DATES'].get('Received') is not None and
            filedict[page_key]['DATES'].get('Accepted') is not None):
            consolidated['DATES'] = filedict[page_key]['DATES']
            break
    else:
        consolidated['DATES'] = {'Received': None, 'Accepted': None}

    # ====== Consolidate MAIN_CONTENT ====== (EXACT same logic as PDF.ipynb with safety checks)
    contents = []
    for i in range(1, len(filedict) + 1):
        page_key = 'page' + str(i)
        if (page_key in filedict and 
            'MAIN_CONTENT' in filedict[page_key] and 
            filedict[page_key]['MAIN_CONTENT'] is not None):
            contents.append(filedict[page_key]['MAIN_CONTENT'])

    # Store as JSON string for SQLite (same as PDF.ipynb)
    consolidated['MAIN_CONTENT'] = json.dumps(contents) if contents else None

    # ====== Consolidate REFERENCE_COUNT ====== (FROM features.ipynb logic with safety checks)
    total_page = len(filedict)
    if total_page <= 2:
        pages_to_consider = [total_page]
    elif 3 <= total_page <= 10:
        pages_to_consider = [total_page - 1, total_page]
    else:  # total_page >= 11
        pages_to_consider = [total_page - 2, total_page - 1, total_page]

    total_refs = 0
    for p in pages_to_consider:
        page_key = 'page' + str(p)
        if (page_key in filedict and 
            'REFERENCE_COUNT' in filedict[page_key] and 
            filedict[page_key]['REFERENCE_COUNT'] is not None):
            if isinstance(filedict[page_key]['REFERENCE_COUNT'], int):
                total_refs += filedict[page_key]['REFERENCE_COUNT']
            elif isinstance(filedict[page_key]['REFERENCE_COUNT'], dict):
                total_refs += sum(
                    v for v in filedict[page_key]['REFERENCE_COUNT'].values() if isinstance(v, int)
                )

    consolidated['REFERENCE_COUNT'] = total_refs

    return consolidated


# EXACT COPY from features.ipynb - Grammar checking with LanguageTool API
# ENHANCED with HIGH CONCURRENCY: 15+ parallel LanguageTool API calls

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp

# EXACT COPY from PDF.ipynb - Feature calculation functions

def word_counter(txt):
    word_count = len(txt.split())
    return word_count

# Date processing
def calculate_date_difference(start_date_str, end_date_str):
    # Parse the date strings into date objects using strptime
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    
    # Calculate the difference between the two dates
    delta = end_date - start_date
    
    # Return the number of days in the difference
    return delta.days

# Flesch Kincaid Calculator
pronouncing_dict = cmudict.dict()

def count_syllables(word):
    word = word.lower()
    if word in pronouncing_dict:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in pronouncing_dict[word]])

    # Fallback regex-based syllable count
    VC = re.compile(r'[aeiouy]+[^aeiouy]*', re.I)
    return len(VC.findall(word))

def compute_fres(txt):
    clean_txt = txt.lower().translate(str.maketrans("", "", string.punctuation))
    word_list = clean_txt.split()
    sentences = sent_tokenize(txt)

    num_sents = len(sentences)
    num_words = len(word_list)
    num_syllables = sum(count_syllables(w) for w in word_list)

    score = 206.835 - 1.015 * (num_words / num_sents) - 84.6 * (num_syllables / num_words)
    return score

# MTLD and HD-D calculation
def lexical_calc(txt):
    lex = LexicalRichness(txt)
    mtld = lex.mtld(threshold=0.80)
    hdd = lex.hdd(draws=42)
    return mtld, hdd

# Lexical Density calculation
stop_words = set(stopwords.words('english'))

def lex_density_calc(txt):
    tokenized = sent_tokenize(txt)
    all_tagged_words = []

    for sentence in tokenized:
        words = word_tokenize(sentence)
        words = [w for w in words if w.lower() not in stop_words]
        tagged_words = nltk.pos_tag(words)
        all_tagged_words.extend(tagged_words)

    content_words = {"NN", "NNS", "NNP", "NNPS",  # Nouns
                    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # Verbs
                    "JJ", "JJR", "JJS",  # Adjectives
                    "RB", "RBR", "RBS"}  # Adverbs

    contentword_count = 0
    for tag in all_tagged_words:
        if tag[1] in content_words:
            contentword_count += 1

    total_words = word_counter(txt)

    lex_density  = contentword_count / total_words
    return lex_density

# HIGH CONCURRENCY Grammar checking with LanguageTool API (15+ parallel calls)
@lru_cache(maxsize=256)  # Increased cache size
def count_issues_cached(text_hash, language="en-US"):
    """Cached version to avoid repeated API calls for same text"""
    return _count_issues_internal(text_hash, language)

def _count_issues_internal(text, language="en-US"):
    """Internal function to make the actual API call"""
    url = "https://api.languagetool.org/v2/check"
    params = {
        'text': text,
        'language': language
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, data=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            return len(result.get('matches', []))
        except requests.exceptions.RequestException as e:
            print(f"Grammar check attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 0.5)  # Reduced backoff for faster retry
            else:
                print("All grammar check attempts failed, returning 0")
                return 0

def count_grammar_issues(text, language="en-US"):
    """Main function with caching and chunking for large texts - EXACT COPY from features.ipynb"""
    if not text or len(text.strip()) == 0:
        return 0
    
    # Chunk large texts to avoid API limits
    max_chunk_size = 10000
    if len(text) <= max_chunk_size:
        return count_issues_cached(text, language)
    
    # Split into chunks and sum results
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    total_issues = 0
    
    for chunk in chunks:
        total_issues += count_issues_cached(chunk, language)
        time.sleep(0.3)  # Reduced rate limiting for faster processing
    
    return total_issues

# NEW: HIGH CONCURRENCY PARALLEL Grammar Checking (15+ concurrent calls)
def count_grammar_issues_parallel(text_chunks, max_workers=15, language="en-US"):
    """
    Process multiple text chunks in parallel with HIGH CONCURRENCY.
    Uses 15+ parallel LanguageTool API calls for maximum speed.
    
    Args:
        text_chunks: List of text chunks to check
        max_workers: Number of parallel LanguageTool API calls (default: 15)
        language: Language code for checking (default: en-US)
    
    Returns:
        Total grammar issues across all chunks
    """
    if not text_chunks:
        return 0
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in text_chunks if chunk and chunk.strip()]
    if not valid_chunks:
        return 0
    
    total_issues = 0
    
    print(f"PARALLEL Grammar checking with {max_workers} parallel LanguageTool API calls...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for parallel processing
        future_to_chunk = {
            executor.submit(count_issues_cached, chunk, language): chunk 
            for chunk in valid_chunks
        }
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_chunk):
            completed_count += 1
            try:
                issues = future.result(timeout=60)
                total_issues += issues
                if completed_count % 5 == 0:  # Progress update every 5 chunks
                    print(f"  Grammar check progress: {completed_count}/{len(valid_chunks)} chunks")
            except Exception as e:
                chunk = future_to_chunk[future]
                print(f"Error checking chunk: {e}")
                # Continue processing other chunks
    
    print(f"SUCCESS: Grammar checking complete: {total_issues} total issues found")
    return total_issues

# NEW: ASYNC HIGH CONCURRENCY Grammar Checking (even faster)
async def count_grammar_issues_async(text_chunks, max_workers=15, language="en-US"):
    """
    ASYNC version with even higher concurrency for maximum speed.
    Uses async HTTP calls with aiohttp for optimal performance.
    
    Args:
        text_chunks: List of text chunks to check
        max_workers: Number of concurrent async calls (default: 15)
        language: Language code for checking
    
    Returns:
        Total grammar issues across all chunks
    """
    if not text_chunks:
        return 0
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in text_chunks if chunk and chunk.strip()]
    if not valid_chunks:
        return 0
    
    print(f"ASYNC Grammar checking with {max_workers} concurrent calls...")
    
    async def check_single_chunk(session, semaphore, chunk):
        async with semaphore:
            url = "https://api.languagetool.org/v2/check"
            data = {'text': chunk, 'language': language}
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with session.post(url, data=data, timeout=30) as response:
                        result = await response.json()
                        return len(result.get('matches', []))
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 + attempt * 0.3)
                    else:
                        print(f"Async grammar check failed after {max_retries} attempts: {e}")
                        return 0
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_workers)
    
    # Process all chunks concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [check_single_chunk(session, semaphore, chunk) for chunk in valid_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sum up results, handling any exceptions
        total_issues = 0
        for result in results:
            if isinstance(result, int):
                total_issues += result
            else:
                print(f"Async grammar check exception: {result}")
    
    print(f"SUCCESS: ASYNC Grammar checking complete: {total_issues} total issues found")
    return total_issues

# SMART: Auto-chunking function for optimal parallel processing
def count_grammar_issues_SMART_PARALLEL(text, max_workers=15, language="en-US", chunk_size=8000):
    """
    SMART parallel grammar checking that automatically chunks text optimally.
    Uses high concurrency (15+ parallel calls) for maximum speed.
    
    Args:
        text: Full text to check
        max_workers: Number of parallel LanguageTool API calls (default: 15)
        language: Language code for checking
        chunk_size: Size of each chunk in characters (default: 8000)
    
    Returns:
        Total grammar issues in the text
    """
    if not text or len(text.strip()) == 0:
        return 0
    
    # For small texts, use simple cached version
    if len(text) <= chunk_size:
        return count_issues_cached(text, language)
    
    # Split into optimal chunks
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    
    if len(chunks) <= 1:
        return count_issues_cached(text, language)
    
    print(f"INFO: Processing {len(chunks)} text chunks with {max_workers} parallel workers")
    
    # Use parallel processing for multiple chunks
    return count_grammar_issues_parallel(chunks, max_workers=max_workers, language=language)




def extract_features_single_document(pdf_file_path, max_openai_workers=25, max_languagetool_workers=20):
    """Backwards compatible wrapper with configurable concurrency"""
    # Enforce maximum limits for API safety
    max_openai_workers = min(max_openai_workers, 25)  # Cap at 25 for OpenAI
    max_languagetool_workers = min(max_languagetool_workers, 20)  # Cap at 20 for LanguageTool
    
    return extract_features_multiprocessed_article(
        pdf_file_path, 
        max_workers=max_openai_workers, 
        max_languagetool_workers=max_languagetool_workers
    )


def extract_features_multiprocessed_article(pdf_file_path, max_workers=25, max_languagetool_workers=25):
    """
    Extract all features from a single PDF document with configurable concurrency.
    This is the main pipeline function that combines all the existing functions.
    
    Args:
        pdf_file_path (str): Path to the PDF file
        max_workers (int): Number of parallel workers for PDF processing (OpenAI calls)
        max_languagetool_workers (int): Number of parallel LanguageTool API calls
    
    Returns:
        dict: Extracted features for the document
    """
    try:
        print(f"📄 Processing: {os.path.basename(pdf_file_path)}")
        
        # Step 1: Process PDF using optimized PDF processing
        print("  • Extracting and cleaning PDF content...")
        file_dict = process_pdf_optimized(pdf_file_path, max_workers=max_workers)
        
        # Step 2: Consolidate data (including reference count)
        print("  • Consolidating content and reference count...")
        consolidated = consolidate_enhanced(file_dict)
        
        # Step 3: Extract main content for feature calculation
        if consolidated['MAIN_CONTENT']:
            try:
                main_content_list = json.loads(consolidated['MAIN_CONTENT'])
                full_text = ' '.join(main_content_list) if main_content_list else ""
            except:
                full_text = consolidated['MAIN_CONTENT'] if consolidated['MAIN_CONTENT'] else ""
        else:
            full_text = ""
        
        if not full_text.strip():
            print(f"  ERROR: No content extracted from {os.path.basename(pdf_file_path)}")
            return None
        
        # Step 4: Calculate features
        print("  • Calculating word count...")
        word_count = word_counter(full_text)
        
        print("  • Calculating readability score...")
        readability = compute_fres(full_text)
        
        print("  • Calculating lexical features...")
        mtld, hdd = lexical_calc(full_text)
        lexical_density = lex_density_calc(full_text)
        
        # Step 5: Grammar checking with high concurrency
        print(f"  • Checking grammar errors with {max_languagetool_workers} LanguageTool workers...")
        grammar_errors = count_grammar_issues_SMART_PARALLEL(full_text, max_workers=max_languagetool_workers)
        
        # Step 6: Calculate review speed (if dates available)
        review_speed = None
        if (consolidated['DATES']['Received'] and 
            consolidated['DATES']['Accepted'] and
            consolidated['DATES']['Received'] != 'None' and
            consolidated['DATES']['Accepted'] != 'None'):
            try:
                review_speed = calculate_date_difference(
                    consolidated['DATES']['Received'],
                    consolidated['DATES']['Accepted']
                )
            except:
                print(f"  WARNING: Could not calculate review speed for {os.path.basename(pdf_file_path)}")
                review_speed = None
        
        # Step 7: Compile features
        features = {
            'file_name': os.path.basename(pdf_file_path),
            'received_date': consolidated['DATES']['Received'],
            'accepted_date': consolidated['DATES']['Accepted'],
            'word_count': word_count,
            'grammar_errors': grammar_errors,
            'review_speed': review_speed,
            'readability': readability,
            'mtld': mtld,
            'hdd': hdd,
            'lexical_density': lexical_density,
            'reference_count': consolidated['REFERENCE_COUNT']
        }
        
        return features
        
    except Exception as e:
        print(f"  ERROR: Error processing {os.path.basename(pdf_file_path)}: {e}")
        return None


# PREDICTION ANALYSIS FUNCTION


def predscan_analyze(features_dict):
    """
    Analyze a document using the trained Gradient Boosting model for predatory journal detection.
    
    Takes the output from extract_features_multiprocessed_article and returns predictions
    with both class prediction (0 or 1) and probability percentages.
    
    Args:
        features_dict (dict): Dictionary containing extracted features from extract_features_multiprocessed_article.
                             Expected keys: word_count, grammar_errors, review_speed, readability, 
                             mtld, hdd, lexical_density, reference_count
    
    Returns:
        dict: Analysis results containing:
            - success (bool): Whether analysis was successful
            - prediction (int): 0 for legitimate, 1 for predatory (only if successful)
            - probability_legitimate (float): Probability percentage for legitimate class (0-100)
            - probability_predatory (float): Probability percentage for predatory class (0-100)
            - confidence (str): Confidence level description
            - error_message (str): Error description (only if unsuccessful)
            - features_used (dict): The 4 features used by the model
    
    Example:
        features = extract_features_multiprocessed_article("document.pdf")
        result = predscan_analyze(features)
        
        if result['success']:
            print(f"Prediction: {'Predatory' if result['prediction'] == 1 else 'Legitimate'}")
            print(f"Confidence: {result['probability_predatory']}% predatory")
    """
    
    try:
        # Step 1: Validate input
        if not features_dict or not isinstance(features_dict, dict):
            return {
                'success': False,
                'error_message': 'Invalid input: features_dict must be a non-empty dictionary'
            }
        
        # Step 2: Check for required features
        required_features = [
            'word_count', 'grammar_errors', 'review_speed', 'readability',
            'mtld', 'hdd', 'lexical_density', 'reference_count'
        ]
        
        missing_features = [f for f in required_features if f not in features_dict]
        if missing_features:
            return {
                'success': False,
                'error_message': f'Missing required features: {", ".join(missing_features)}'
            }
        
        # Step 3: Handle missing review_speed (common issue)
        # If review_speed is None, we need to decide how to handle it
        if features_dict['review_speed'] is None:
            print("  WARNING: Review speed unavailable (missing publication dates), using median value")
            # Use a reasonable default based on academic publishing (approximately 90 days)
            features_dict = features_dict.copy()  # Don't modify original
            features_dict['review_speed'] = 90.0
        
        # Step 4: Validate feature values
        for feature, value in features_dict.items():
            if feature in required_features and value is None:
                return {
                    'success': False,
                    'error_message': f'Feature {feature} has None value and cannot be processed'
                }
        
        # Step 5: Load preprocessing scaler
        try:
            scaler = joblib.load('preprocessing_scaler.pkl')
            print("  Preprocessing scaler loaded successfully")
        except FileNotFoundError:
            return {
                'success': False,
                'error_message': 'Preprocessing scaler file not found: preprocessing_scaler.pkl'
            }
        except Exception as e:
            return {
                'success': False,
                'error_message': f'Error loading preprocessing scaler: {str(e)}'
            }
        
        # Step 6: Load trained model
        try:
            model = joblib.load('final_gradient_boosting.pkl')
            print("  Gradient Boosting model loaded successfully")
        except FileNotFoundError:
            return {
                'success': False,
                'error_message': 'Model file not found: final_gradient_boosting.pkl'
            }
        except Exception as e:
            return {
                'success': False,
                'error_message': f'Error loading model: {str(e)}'
            }
        
        # Step 7: Prepare features in correct order for preprocessing
        # Based on preprocessing_info.json, the scaler expects 8 features in this order:
        preprocessing_feature_order = [
            'word_count', 'grammar_errors', 'review_speed', 'readability',
            'mtld', 'hdd', 'lexical_density', 'reference_count'
        ]
        
        # Extract features in the correct order
        feature_values = []
        for feature_name in preprocessing_feature_order:
            value = features_dict[feature_name]
            # Ensure numeric values
            try:
                numeric_value = float(value)
                feature_values.append(numeric_value)
            except (ValueError, TypeError):
                return {
                    'success': False,
                    'error_message': f'Feature {feature_name} has invalid numeric value: {value}'
                }
        
        # Convert to numpy array for sklearn
        features_array = np.array(feature_values).reshape(1, -1)
        
        # Step 8: Apply preprocessing scaler
        try:
            
            # Verify scaler is callable and has transform method
            if not hasattr(scaler, 'transform'):
                # Try to handle different scaler formats
                return {
                    'success': False,
                    'error_message': f'Loaded scaler object ({type(scaler)}) does not have transform method. Available methods: {[attr for attr in dir(scaler) if not attr.startswith("_")]}'
                }
            else:
                # Standard scaler transform
                scaled_features = scaler.transform(features_array)
                print("  Features scaled successfully")
                
        except Exception as e:
            return {
                'success': False,
                'error_message': f'Error applying preprocessing scaler: {str(e)}. Scaler type: {type(scaler)}, Available methods: {[attr for attr in dir(scaler) if not attr.startswith("_")]}'
            }
        
        # Step 9: Select only the 5 features used by the final model
        # Based on actual model feature names: ['word_count', 'grammar_errors', 'review_speed', 'lexical_density', 'reference_count']
        model_feature_indices = {
            'word_count': 0,      # Index in preprocessing_feature_order
            'grammar_errors': 1,  # Index in preprocessing_feature_order
            'review_speed': 2,    # Index in preprocessing_feature_order  
            'lexical_density': 6, # Index in preprocessing_feature_order
            'reference_count': 7  # Index in preprocessing_feature_order
        }
        
        try:
            # Extract the 5 features in the order expected by the model
            model_features = scaled_features[:, [
                model_feature_indices['word_count'],
                model_feature_indices['grammar_errors'],
                model_feature_indices['review_speed'], 
                model_feature_indices['lexical_density'],
                model_feature_indices['reference_count']
            ]]
            
        except Exception as e:
            return {
                'success': False,
                'error_message': f'Error selecting model features from scaled data: {str(e)}. Scaled features shape: {scaled_features.shape if scaled_features is not None else "None"}'
            }
        
        # Step 10: Make prediction
        try:
            prediction = model.predict(model_features)[0]
            probabilities = model.predict_proba(model_features)[0]
            print("  Prediction completed successfully")
        except Exception as e:
            return {
                'success': False,
                'error_message': f'Error making prediction: {str(e)}'
            }
        
        # Step 11: Calculate probability percentages
        prob_legitimate = probabilities[0] * 100  # Class 0 (legitimate)
        prob_predatory = probabilities[1] * 100   # Class 1 (predatory)
        
        # Step 12: Determine confidence level
        max_prob = max(prob_legitimate, prob_predatory)
        if max_prob >= 80:
            confidence = "High"
        elif max_prob >= 65:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Step 13: Prepare features used by model for transparency
        # Include all extracted features to support frontend display
        features_used = {
            'word_count': features_dict['word_count'],
            'grammar_errors': features_dict['grammar_errors'], # Used by model
            'review_speed': features_dict['review_speed'],
            'lexical_density': features_dict['lexical_density'],
            'reference_count': features_dict['reference_count'], # Used by model
            # Add the three additional features for frontend display
            'readability': features_dict['readability'],
            'mtld': features_dict['mtld'],
            'hdd': features_dict['hdd']
        }
        
        # Step 14: Return comprehensive results
        return {
            'success': True,
            'prediction': int(prediction),
            'probability_legitimate': round(prob_legitimate, 2),
            'probability_predatory': round(prob_predatory, 2),
            'confidence': confidence,
            'features_used': features_used,
            'model_info': {
                'model_type': 'Gradient Boosting Classifier',
                'features_count': 5,
                'preprocessing': 'RobustScaler'
            }
        }
        
    except Exception as e:
        # Catch any unexpected errors
        return {
            'success': False,
            'error_message': f'Unexpected error during analysis: {str(e)}'
        }


def debug_scaler_loading():
    """
    Debug function to investigate the scaler loading issue.
    Call this function to get detailed information about the scaler file.
    """
    print("DEBUG: Debugging scaler loading...")
    
    # Check if file exists
    import os
    if not os.path.exists('preprocessing_scaler.pkl'):
        print("❌ preprocessing_scaler.pkl file not found")
        return
    
    print("✅ preprocessing_scaler.pkl file found")
    print(f"📊 File size: {os.path.getsize('preprocessing_scaler.pkl')} bytes")
    
    # Try different loading methods
    loading_methods = [
        ("joblib.load", lambda: joblib.load('preprocessing_scaler.pkl')),
        ("pickle.load", lambda: pickle.load(open('preprocessing_scaler.pkl', 'rb')))
    ]
    
    for method_name, loader in loading_methods:
        try:
            print(f"\n🔄 Trying {method_name}...")
            scaler = loader()
            print(f"  ✅ Loaded successfully with {method_name}")
            print(f"  📊 Type: {type(scaler)}")
            print(f"  📊 Has transform method: {hasattr(scaler, 'transform')}")
            
            if isinstance(scaler, dict):
                print(f"  📊 Dictionary keys: {list(scaler.keys())}")
            else:
                print(f"  📊 Available methods: {[attr for attr in dir(scaler) if not attr.startswith('_')]}")
                
                # Try to get scaler attributes
                if hasattr(scaler, 'center_'):
                    print(f"  📊 Has center_ attribute: {hasattr(scaler, 'center_')}")
                if hasattr(scaler, 'scale_'):
                    print(f"  📊 Has scale_ attribute: {hasattr(scaler, 'scale_')}")
                if hasattr(scaler, 'mean_'):
                    print(f"  📊 Has mean_ attribute: {hasattr(scaler, 'mean_')}")
            
            # Test transform if available
            if hasattr(scaler, 'transform'):
                try:
                    test_data = np.array([[1000, 5, 90, 50, 100, 0.5, 0.3, 20]]).reshape(1, -1)
                    result = scaler.transform(test_data)
                    print(f"  ✅ Transform test successful, output shape: {result.shape}")
                except Exception as e:
                    print(f"  ❌ Transform test failed: {e}")
            
        except Exception as e:
            print(f"  ❌ Failed with {method_name}: {e}")
    
    print("\n🏁 Debug complete")
