'''
Given a code file, count "size" as follows:
- Ignore everything after "TESTS" (case-sensitive)
- Triple-quote strings don't count
- Comments don't count
- Any consecutive sequence of alphanumeric characters and underscores counts as one "word"
- No other characters count as words
'''

def code_size(code:str) -> int:
    import re
    # Ignore everything after "TESTS"
    code = re.sub(r'TESTS.*', '', code, flags=re.DOTALL)
    # Remove triple-quote strings
    code = re.sub(r'\'\'\'[\s\S]*?\'\'\'', '', code)
    # Remove comments
    code = re.sub(r'#.*', '', code)
    # Count words
    return len(re.findall(r'\w+', code))

def file_size(path:str) -> int:
    with open(path, 'r') as f:
        return code_size(f.read())

print(file_size('elo_optimizer.py')) # Original was 590