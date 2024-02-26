# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Collect data for building tables to publish via mkdocs
"""

from pathlib import Path

import yaml

from ultralytics.utils import ROOT as PACKAGE_DIR

GITHUB_REPO = "ultralytics/ultralytics"

cfg_sections = [
    "Train settings",
    "Segmentation train settings",
    "Classification train settings",
    "Val/Test settings",
    "Predict settings",
    "Visualize settings",
    "Export settings",
    "Hyperparameters",
    "Augmentation",
    "Segmentation Augmentations",
    "Classification Augmentations",
    "Custom config.yaml",
    "Tracker",
]

default_cfg = PACKAGE_DIR / "cfg/default.yaml"
cfg = yaml.safe_load(default_cfg.read_text("utf-8"))
# text = [l.strip() for l in default_cfg.read_text("utf-8") if any(l)]
txt = [l.strip(" -") for l in default_cfg.read_text("utf-8").splitlines() if any(l)]

def n_section(N:int):
    return "#" * N + " "

def section_depth(section:str):
    return section.split(" ")[0].count("#")

def find_sections(txt_data:list[str], max_repeats:int=2) -> list[tuple[int,str]]:
    section_starts = [("#"*i + " ") for i in range(1,max_repeats + 1)]
    sections = list()
    for ss in section_starts:
        # _ = [sections.append((li, l.strip(" -"))) for li,l in enumerate(txt_data) if l.startswith(ss)]
        _ = [sections.append(l.strip(" -")) for l in txt_data if l.startswith(ss)]
    # return sorted(sections)
    return sections

def section_span(
        section:tuple[int,str],
        section_list:list[tuple[int,str]],
        lvl:int=1
    ) -> tuple[int,int]:
    # filter(lambda x, section: section in x, section_list)
    idx = section_list.index(section) + 1
    next_at = "#"*lvl + " "
    span = 0
    for s in section_list[idx:]:
        line_num, title = s
        if title.startswith(next_at):
            span = line_num
            break
    return (section[0], span)

def has_subsections(txt_data:list[str], span:tuple[int,int], next_lvl:int=2):
    s = "#" * next_lvl + " "
    s0, s1 = span
    return any(list(filter(lambda x: x.startswith(s), txt_data[s0:s1])))

def text_2_key(line:str):
    return line.strip("# -") + ": "

def extract_type(line:str):
    l,r = line.find("("), line.find(")")
    l_type = line[l:r].strip(")(")
    line = line[r + 1:].strip()
    for r in [("|","or"), ("optional",""), (", ","")]:
        l_type = l_type.replace(*r)
    return l_type, line

sections = list(filter(lambda x: x.strip("# ") in cfg_sections, find_sections(txt)))

tables = list()
for s in sections:
    section_text = text_2_key(s)
    idx = txt.index(s)
    is_sub = False
    sub_d = 0
    for line in txt[idx + 1:]:
        n = section_depth(line)
        if not n: # section entry
            entry, comment = line.split("# ")
            comment = comment.split("http")[0].replace(":", " -") # drop links and escape colons
            v_type, comment = extract_type(comment)
            k, v = entry.split(": ")
            if not is_sub:
                section_text += f"\n  {k}: \n    value: {v}\n    type: {v_type}\n    desc: {comment}"
            else:
                indent = ' ' * sub_d * 2
                section_text += f"\n{indent}{k}: \n{indent}value: {v}\n{indent}type: {v_type}\n{indent}desc: {comment}"
        elif n == 1: # next-section
            is_sub = False
            break
        elif n >= 2: # sub-section
            is_sub = True
            sub_d = n
            section_text += "\n" + (" " * (n - 1) * 2) + text_2_key(line)
    tables.append(yaml.safe_load(section_text))

