"""
Convert EE627 .py files to .ipynb (Jupyter/Colab notebooks).
Splits code into logical cells and converts comment blocks to markdown.
"""
import json
import re
import os

def source_to_lines(source):
    """Convert a source string to a list of lines for .ipynb format.

    Each line must end with '\\n' except the last one.
    This is the standard Jupyter notebook source format.
    """
    lines = source.split('\n')
    # Remove trailing empty lines
    while lines and lines[-1].strip() == '':
        lines.pop()
    if not lines:
        return []
    # Add \n to every line except the last
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]


def make_notebook(cells):
    """Create a valid .ipynb notebook structure."""
    nb_cells = []
    for cell_type, source in cells:
        if not source.strip():
            continue
        src_lines = source_to_lines(source)
        if not src_lines:
            continue
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": src_lines,
        }
        if cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        nb_cells.append(cell)

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "colab": {
                "provenance": []
            }
        },
        "cells": nb_cells
    }
    return notebook


def extract_docstring(lines):
    """Extract the top-level triple-quoted docstring."""
    if not lines or not lines[0].strip().startswith('"""'):
        return None, lines

    docstring_lines = []
    in_docstring = True
    end_idx = 0

    for i, line in enumerate(lines):
        if i == 0:
            docstring_lines.append(line)
            if line.strip().endswith('"""') and len(line.strip()) > 3:
                end_idx = i + 1
                break
            continue
        docstring_lines.append(line)
        if '"""' in line:
            end_idx = i + 1
            break

    content = '\n'.join(docstring_lines)
    # Remove the triple quotes
    content = content.strip()
    if content.startswith('"""'):
        content = content[3:]
    if content.endswith('"""'):
        content = content[:-3]
    content = content.strip()

    return content, lines[end_idx:]


def is_section_separator(line):
    """Check if a line is a section separator like # ====== or # ------"""
    stripped = line.strip()
    if not stripped.startswith('#'):
        return False
    after_hash = stripped[1:].strip()
    if len(after_hash) >= 10 and all(c in '=-~' for c in after_hash):
        return True
    return False


def is_comment_line(line):
    """Check if a line is a comment."""
    return line.strip().startswith('#')


def is_empty_line(line):
    """Check if a line is empty or whitespace only."""
    return line.strip() == ''


def parse_sections(lines):
    """Parse the file into sections based on # ==== separators."""
    sections = []
    current_section = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect section header: # ====...  followed by # N. TITLE or # TITLE
        if is_section_separator(line):
            # Save previous section
            if current_section:
                sections.append(current_section)
                current_section = []
            current_section.append(line)
        else:
            current_section.append(line)
        i += 1

    if current_section:
        sections.append(current_section)

    return sections


def section_to_cells(section_lines):
    """Convert a section into one or more cells (markdown + code)."""
    cells = []

    # Check if section starts with separator
    if not section_lines:
        return cells

    # Collect leading comment block (section header + description)
    comment_block = []
    code_block = []
    in_header = True
    found_separator = False
    separator_count = 0

    i = 0
    while i < len(section_lines):
        line = section_lines[i]

        if in_header:
            if is_section_separator(line):
                separator_count += 1
                comment_block.append(line)
                # After 2 separators (opening and closing of header), or if next
                # line is code, switch to looking for code
                if separator_count >= 2:
                    # Continue collecting comments after the header block
                    i += 1
                    continue
            elif is_comment_line(line) or is_empty_line(line):
                comment_block.append(line)
            else:
                # Found code - switch mode
                in_header = False
                code_block.append(line)
        else:
            # Check for INTERPRETATION or large comment blocks mid-section
            if is_comment_line(line) and not code_block[-1:] or is_empty_line(line):
                code_block.append(line)
            else:
                code_block.append(line)
        i += 1

    # Now split: header comments -> markdown, rest -> code
    # But we need to be smarter: find where code actually starts in the section
    comment_part = []
    code_part = []
    found_code = False

    for line in section_lines:
        if not found_code:
            if is_section_separator(line) or is_comment_line(line) or is_empty_line(line):
                comment_part.append(line)
            else:
                found_code = True
                code_part.append(line)
        else:
            code_part.append(line)

    # Convert comment part to markdown
    if comment_part:
        md_text = comments_to_markdown(comment_part)
        if md_text.strip():
            cells.append(["markdown", md_text])

    # Now split the code part: look for large trailing comment blocks
    # (INTERPRETATION sections) and split them out as markdown
    if code_part:
        code_chunks = split_code_and_comments(code_part)
        cells.extend(code_chunks)

    return cells


def comments_to_markdown(comment_lines):
    """Convert a block of Python comments to markdown text."""
    md_lines = []
    for line in comment_lines:
        stripped = line.strip()
        if is_section_separator(line):
            continue  # Skip separator lines
        elif stripped.startswith('# '):
            md_lines.append(stripped[2:])
        elif stripped == '#':
            md_lines.append('')
        elif stripped.startswith('#'):
            md_lines.append(stripped[1:])
        elif stripped == '':
            md_lines.append('')
        else:
            md_lines.append(stripped)

    # Clean up leading/trailing empty lines
    while md_lines and md_lines[0].strip() == '':
        md_lines.pop(0)
    while md_lines and md_lines[-1].strip() == '':
        md_lines.pop()

    text = '\n'.join(md_lines)

    # Detect section titles (e.g., "1. LOAD DATA" or "SETUP")
    lines = text.split('\n')
    if lines:
        first = lines[0].strip()
        # Check if first line looks like a section title
        if (re.match(r'^\d+\.?\s+[A-Z]', first) or
            (first.isupper() and len(first) < 80 and not first.startswith('-'))):
            lines[0] = f'## {first}'
            text = '\n'.join(lines)

    return text


def split_code_and_comments(code_lines):
    """Split code lines into code cells and markdown cells for large comment blocks."""
    chunks = []
    current_code = []
    current_comments = []
    in_comment_block = False
    comment_block_start = -1

    i = 0
    while i < len(code_lines):
        line = code_lines[i]

        if not in_comment_block:
            if is_comment_line(line) or is_empty_line(line):
                # Check if this is the start of a large comment block (5+ comment lines)
                lookahead_comments = 0
                j = i
                while j < len(code_lines) and (is_comment_line(code_lines[j]) or is_empty_line(code_lines[j])):
                    if is_comment_line(code_lines[j]):
                        lookahead_comments += 1
                    j += 1

                if lookahead_comments >= 6 and j == len(code_lines):
                    # Large trailing comment block -> make it markdown
                    if current_code:
                        chunks.append(["code", '\n'.join(current_code)])
                        current_code = []
                    # Collect all remaining as comments
                    remaining = code_lines[i:]
                    md_text = comments_to_markdown(remaining)
                    if md_text.strip():
                        chunks.append(["markdown", md_text])
                    break
                else:
                    current_code.append(line)
            else:
                current_code.append(line)
        i += 1

    if current_code:
        chunks.append(["code", '\n'.join(current_code)])

    return chunks


def detect_data_files(py_path):
    """Scan the .py file for pd.read_excel/csv/stata calls and return filenames."""
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Match pd.read_excel("..."), pd.read_csv("..."), pd.read_stata("...")
    pattern = r'pd\.read_(?:excel|csv|stata)\(["\']([^"\']+)["\']\)'
    matches = re.findall(pattern, content)
    # Return unique filenames in order
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def make_upload_cell(data_files):
    """Create a Colab file upload code cell for the required data files."""
    files_list = ', '.join(f'"{f}"' for f in data_files)
    code = (
        "# --- Google Colab: Upload Data Files ---\n"
        "# Run this cell to upload the required data file(s).\n"
        "# You can also mount Google Drive instead (see alternative below).\n"
        "\n"
        "import os\n"
        "\n"
        f"required_files = [{files_list}]\n"
        "missing = [f for f in required_files if not os.path.exists(f)]\n"
        "\n"
        "if missing:\n"
        "    try:\n"
        "        from google.colab import files\n"
        "        print(f'Please upload: {missing}')\n"
        "        uploaded = files.upload()\n"
        "        print(f'Uploaded: {list(uploaded.keys())}')\n"
        "    except ImportError:\n"
        "        print('Not running on Colab. Make sure these files are in the working directory:')\n"
        "        print(missing)\n"
        "else:\n"
        "    print(f'All data files found: {required_files}')\n"
        "\n"
        "# Alternative: Mount Google Drive (uncomment below)\n"
        "# from google.colab import drive\n"
        "# drive.mount('/content/drive')\n"
        "# %cd /content/drive/MyDrive/EE627"
    )
    return code


def convert_py_to_ipynb(py_path, ipynb_path, chapter_info):
    """Convert a .py file to .ipynb with smart cell splitting."""
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    cells = []

    # 0. Detect required data files
    data_files = detect_data_files(py_path)

    # 1. Extract top docstring -> markdown title cell
    docstring, remaining_lines = extract_docstring(lines)

    if docstring:
        # Create a nice title markdown cell
        ch_num, ch_title = chapter_info
        title_md = f"# {ch_num}: {ch_title}\n\n"
        title_md += f"**EE627 Microeconometrics** | Asst. Prof. Dr. Tiraphap Fakthong | Thammasat University\n\n"
        title_md += "---\n\n"

        # Extract overview from docstring
        overview_lines = docstring.split('\n')
        # Find the content between the header and the end
        in_content = False
        content_lines = []
        for dl in overview_lines:
            stripped = dl.strip()
            if all(c in '=-' for c in stripped) and len(stripped) > 5:
                in_content = not in_content
                continue
            if stripped.startswith('Instructor:') or stripped.startswith('Thammasat'):
                continue
            if stripped.startswith('Dataset:') or stripped.startswith('File:') or stripped.startswith('This script replicates'):
                content_lines.append(stripped)
                continue
            if stripped.startswith('Required packages:'):
                content_lines.append(f'\n**{stripped}**')
                continue
            if stripped.startswith('pip install'):
                content_lines.append(f'```\n{stripped}\n```')
                continue
            if stripped.startswith('Designed to run'):
                content_lines.append(f'*{stripped}*')
                continue
            content_lines.append(dl.rstrip())

        title_md += '\n'.join(content_lines)
        cells.append(["markdown", title_md])

    # 1b. Add data upload cell if this chapter requires data files
    if data_files:
        files_str = ', '.join(f'`{f}`' for f in data_files)
        cells.append(["markdown",
            f"## Upload Data Files\n\n"
            f"This notebook requires the following data file(s): {files_str}\n\n"
            f"Run the cell below to upload them (or mount Google Drive)."])
        cells.append(["code", make_upload_cell(data_files)])

    # 2. Parse remaining lines into sections
    sections = parse_sections(remaining_lines)

    # 3. Convert each section to cells
    for section in sections:
        section_cells = section_to_cells(section)
        cells.extend(section_cells)

    # Filter empty cells
    cells = [c for c in cells if c[1].strip()]

    # Build notebook
    notebook = make_notebook(cells)

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    return len(notebook['cells'])


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    chapters = [
        ('Chap1_OLS.py', 'Chapter 1', 'OLS Regression, Specification Analysis, and Prediction'),
        ('Chap2_MonteCarlo.py', 'Chapter 2', 'Monte Carlo Simulation Methods'),
        ('Chap3_HetSUR.py', 'Chapter 3', 'Heteroskedasticity, SUR, and Survey Data'),
        ('Chap4_IV.py', 'Chapter 4', 'Instrumental Variables Estimation'),
        ('Chap5_Quantile.py', 'Chapter 5', 'Quantile Regression and Count Data Models'),
        ('Chap6_Panel.py', 'Chapter 6', 'Panel Data - Linear and Nonlinear Models'),
        ('Chap7_RD.py', 'Chapter 7', 'Regression Discontinuity Designs'),
        ('Chap8_DID.py', 'Chapter 8', 'Difference-in-Differences'),
    ]

    print("Converting .py files to .ipynb notebooks...")
    print("=" * 60)

    for filename, ch_num, ch_title in chapters:
        py_path = os.path.join(base_dir, filename)
        ipynb_name = filename.replace('.py', '.ipynb')
        ipynb_path = os.path.join(base_dir, ipynb_name)

        if not os.path.exists(py_path):
            print(f"  SKIP: {filename} not found")
            continue

        data_files = detect_data_files(py_path)
        n_cells = convert_py_to_ipynb(py_path, ipynb_path, (ch_num, ch_title))
        data_info = f"  data: {', '.join(data_files)}" if data_files else "  (no data needed)"
        print(f"  {ipynb_name:30s} -> {n_cells} cells{data_info}")

    print("=" * 60)
    print("Done! Upload the .ipynb files to Google Colab.")


if __name__ == '__main__':
    main()
