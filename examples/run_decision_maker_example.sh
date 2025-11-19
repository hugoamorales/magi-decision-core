#!/bin/bash
# Example script to run the Decision Maker Pipeline

echo "====================================================="
echo "  MAGI Decision Maker Pipeline - Example Run"
echo "====================================================="
echo ""

# Navigate to examples directory
cd "$(dirname "$0")"

# Example problem scenarios - choose one
EXAMPLE=1

case $EXAMPLE in
  1)
    PROBLEM="Our startup has limited resources. Should we focus on building more features for existing customers or invest in marketing to acquire new customers?"
    ;;
  2)
    PROBLEM="We need to decide between migrating our monolithic application to microservices or continuing to scale the current architecture. What's the best approach?"
    ;;
  3)
    PROBLEM="Our team is experiencing burnout. How should we restructure our work processes and team organization to improve wellbeing and productivity?"
    ;;
  *)
    PROBLEM="How can we improve our product's user retention rate?"
    ;;
esac

echo "Running Decision Maker Pipeline..."
echo "Problem: $PROBLEM"
echo ""

python pipeline_runner.py \
  --pipeline pipelines/decision_maker_pipeline.yml \
  --problem "$PROBLEM" \
  --output-dir ./results/decision_maker

echo ""
echo "====================================================="
echo "  Pipeline Complete!"
echo "====================================================="
echo ""
echo "Check the results directory for outputs:"
echo "  - JSON file: Raw data"
echo "  - Markdown file: Readable format"
echo "  - HTML file: Interactive UI (open in browser)"
echo ""
echo "To run with a custom problem:"
echo "  python pipeline_runner.py \\"
echo "    --pipeline pipelines/decision_maker_pipeline.yml \\"
echo "    --problem \"Your problem here\" \\"
echo "    --output-dir ./results/decision_maker"
echo ""
