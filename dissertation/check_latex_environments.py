#!/usr/bin/env python3
import re
import os

def check_latex_environments(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return ["Could not read file"]

    # Find all \begin{} and \end{} statements
    begins = re.findall(r'\\begin\{([^}]+)\}', content)
    ends = re.findall(r'\\end\{([^}]+)\}', content)

    begin_count = {}
    end_count = {}

    for env in begins:
        begin_count[env] = begin_count.get(env, 0) + 1

    for env in ends:
        end_count[env] = end_count.get(env, 0) + 1

    issues = []
    all_envs = set(begin_count.keys()) | set(end_count.keys())

    for env in all_envs:
        b_count = begin_count.get(env, 0)
        e_count = end_count.get(env, 0)
        if b_count != e_count:
            issues.append(f'{env}: {b_count} begins, {e_count} ends')

    return issues

def check_common_latex_issues(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return ["Could not read file"]

    issues = []

    # Check for unclosed braces
    open_braces = content.count('{')
    close_braces = content.count('}')
    if open_braces != close_braces:
        issues.append(f"Brace mismatch: {open_braces} open, {close_braces} close")

    # Check for missing $ for math mode
    single_dollar = content.count('$')
    if single_dollar % 2 != 0:
        issues.append("Odd number of $ symbols - possible unclosed math mode")

    # Check for common problematic sequences
    if '\\\\' in content and not re.search(r'\\\\(?:\[[^\]]*\])?(?:\s|$)', content):
        issues.append("Possible incorrect \\\\ usage")

    return issues

# Check all .tex files
tex_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.tex'):
            tex_files.append(os.path.join(root, file))

print('Environment balance check:')
print('=' * 50)
for tex_file in tex_files:
    env_issues = check_latex_environments(tex_file)
    syntax_issues = check_common_latex_issues(tex_file)

    all_issues = env_issues + syntax_issues

    if all_issues:
        print(f'{tex_file}:')
        for issue in all_issues:
            print(f'  - {issue}')
    else:
        print(f'{tex_file}: OK')