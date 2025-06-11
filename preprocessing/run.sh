#!/bin/bash
set -e

python preprocess.py --dataset_name bug_issue
python preprocess.py --dataset_name closed_question
python preprocess.py --dataset_name comment_type --variant java
python preprocess.py --dataset_name comment_type --variant python
python preprocess.py --dataset_name comment_type --variant pharo
python preprocess.py --dataset_name commit_intent
python preprocess.py --dataset_name incivility
python preprocess.py --dataset_name issue_type
python preprocess.py --dataset_name post_api
python preprocess.py --dataset_name post_tag
python preprocess.py --dataset_name question_quality
python preprocess.py --dataset_name requirement_type
python preprocess.py --dataset_name review_aspect
python preprocess.py --dataset_name sentiment
python preprocess.py --dataset_name smell_doc
python preprocess.py --dataset_name story_points
python preprocess.py --dataset_name tone_bearing
