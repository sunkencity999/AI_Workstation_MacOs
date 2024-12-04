#!/bin/bash
export PYTHONPATH=/Users/christopher.bradford/mac_computer_use/local_shellgpt_llm/venv/lib/python3.13/site-packages:/Users/christopher.bradford/local_shellgpt_llm
source /Users/christopher.bradford/local_shellgpt_llm/venv/bin/activate
python3 -m pytest tests/ -v
