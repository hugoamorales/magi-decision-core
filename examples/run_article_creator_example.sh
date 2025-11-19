#!/bin/bash
# Example script to run the Article Creator Pipeline

echo "====================================================="
echo "  MAGI Article Creator Pipeline - Example Run"
echo "====================================================="
echo ""

# Navigate to examples directory
cd "$(dirname "$0")"

# Example article scenarios - choose one
EXAMPLE=1

case $EXAMPLE in
  1)
    IDEA="The Role of AI in Modern Software Development"
    AUDIENCE="Software developers and engineering managers"
    TONES="Professional, Educational, Technical"
    LENGTH="Medium (800-1500 words)"
    ;;
  2)
    IDEA="Building a Sustainable Remote Work Culture"
    AUDIENCE="Business leaders and HR professionals"
    TONES="Professional, Inspirational, Conversational"
    LENGTH="Long (1500-2500 words)"
    ;;
  3)
    IDEA="Introduction to Quantum Computing for Beginners"
    AUDIENCE="Tech enthusiasts with basic programming knowledge"
    TONES="Educational, Conversational, Professional"
    LENGTH="Medium (800-1500 words)"
    ;;
  *)
    IDEA="The Future of Renewable Energy"
    AUDIENCE="General public interested in sustainability"
    TONES="Educational, Inspirational, Professional"
    LENGTH="Medium (800-1500 words)"
    ;;
esac

echo "Running Article Creator Pipeline..."
echo "Topic: $IDEA"
echo "Audience: $AUDIENCE"
echo "Tones: $TONES"
echo "Length: $LENGTH"
echo ""

python pipeline_runner.py \
  --pipeline pipelines/article_creator_pipeline.yml \
  --idea "$IDEA" \
  --audience "$AUDIENCE" \
  --tones "$TONES" \
  --length "$LENGTH" \
  --output-dir ./results/article_creator

echo ""
echo "====================================================="
echo "  Pipeline Complete!"
echo "====================================================="
echo ""
echo "Check the results directory for outputs:"
echo "  - JSON file: Raw data with all drafts"
echo "  - Markdown file: All agent drafts side-by-side"
echo "  - HTML file: Interactive UI with draft comparison (open in browser)"
echo ""
echo "To run with a custom article:"
echo "  python pipeline_runner.py \\"
echo "    --pipeline pipelines/article_creator_pipeline.yml \\"
echo "    --idea \"Your topic here\" \\"
echo "    --audience \"Your target audience\" \\"
echo "    --tones \"Tone1, Tone2, Tone3\" \\"
echo "    --length \"Medium (800-1500 words)\" \\"
echo "    --output-dir ./results/article_creator"
echo ""
