#!/usr/bin/env python3
import re
import os

def fix_textbf_colons(file_path):
    """Fix textbf entries that are missing closing braces before colons"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return False, "Could not read file"

    original_content = content

    # Fix pattern: \textbf{Text: Rest of text (missing closing brace)
    # Replace with: \textbf{Text}: Rest of text
    pattern = r'\\textbf\{([^}]*?):\s*([^}]*?)(?=\\|\n|\s\s)'

    def replace_func(match):
        text_before_colon = match.group(1).strip()
        text_after_colon = match.group(2).strip()
        return f'\\textbf{{{text_before_colon}}}: {text_after_colon}'

    content = re.sub(pattern, replace_func, content)

    # Fix itemize entries with missing braces
    content = re.sub(r'\\item\s+\\textbf\{([^}]+?):\s+([^\\]+)', r'\\item \\textbf{\1}: \2', content)

    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed"
        except:
            return False, "Could not write file"
    else:
        return False, "No changes needed"

def fix_common_brace_issues(file_path):
    """Fix other common brace issues"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return False, "Could not read file"

    original_content = content

    # Fix citations that might have lost braces
    content = re.sub(r'\\cite\{([^}]+)\s+([^}]*)\}', r'\\cite{\1 \2}', content)

    # Look for other common patterns where braces might be missing
    # This is more complex and would need case-by-case analysis

    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed"
        except:
            return False, "Could not write file"
    else:
        return False, "No changes needed"

# Process all .tex files
tex_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.tex'):
            tex_files.append(os.path.join(root, file))

print('Fixing LaTeX brace issues:')
print('=' * 50)

for tex_file in tex_files:
    # Skip the check file itself
    if 'check_latex_environments.py' in tex_file or 'fix_latex_braces.py' in tex_file:
        continue

    fixed, message = fix_textbf_colons(tex_file)
    if fixed:
        print(f'{tex_file}: {message}')

    fixed2, message2 = fix_common_brace_issues(tex_file)
    if fixed2:
        print(f'{tex_file}: {message2}')

print('\nDone. Re-run the environment check to see if issues are resolved.')