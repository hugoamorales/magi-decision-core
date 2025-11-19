# Quick Start Guide - MAGI Pipelines

Get started with the MAGI Decision Core pipelines in under 5 minutes!

## ⚡ Fast Track

### 1. Set Up (One-time)

```bash
# Install dependencies
pip install -r ../requirements.txt

# Set up API keys
cp ../.env_template ../.env
# Edit .env and add your API keys
```

### 2. Run Examples

**Decision Maker Pipeline:**
```bash
./run_decision_maker_example.sh
```

**Article Creator Pipeline:**
```bash
./run_article_creator_example.sh
```

### 3. View Results

Open the generated HTML file in your browser:
```bash
# The script will tell you the exact path, e.g.:
# results/decision_maker/decision_maker_20250119_143022.html
```

## 🎯 What You'll See

### Interactive UI Features:
- ✅ Side-by-side agent outputs
- ✅ Color-coded perspectives (Blue=Analytical, Red=Human-centric, Green=Creative)
- ✅ Confidence meters for each proposal
- ✅ Stage progress tracking
- ✅ Consensus indicators
- ✅ Expandable/collapsible sections

## 📝 Custom Runs

### Decision Maker:
```bash
python pipeline_runner.py \
  --pipeline pipelines/decision_maker_pipeline.yml \
  --problem "Your decision or problem here" \
  --output-dir ./results/decision_maker
```

### Article Creator:
```bash
python pipeline_runner.py \
  --pipeline pipelines/article_creator_pipeline.yml \
  --idea "Your article topic" \
  --audience "Your target readers" \
  --tones "Professional, Educational" \
  --output-dir ./results/article_creator
```

## 🎨 Understanding Agent Outputs

When you view the results, you'll see three perspectives:

### 🔵 Melchior (Blue)
- Analytical and data-driven
- Focuses on efficiency and scalability
- Best for technical insights

### 🔴 Balthasar (Red)
- Human-centric and safety-focused
- Considers user experience and ethics
- Best for accessibility and empathy

### 🟢 Casper (Green)
- Creative and innovative
- Explores novel approaches
- Best for fresh perspectives

## 📊 Output Files Explained

After each run, you get 3 files:

1. **`.json`** - Complete raw data (for programmatic use)
2. **`.md`** - Markdown format (for reading/sharing)
3. **`.html`** - Interactive UI (for visualization) ⭐ **Open this!**

## 🚀 Next Steps

1. ✅ Run both example scripts
2. ✅ Open the HTML files in your browser
3. ✅ Compare agent outputs side-by-side
4. ✅ Try with your own problems/topics
5. ✅ Read [README.md](README.md) for advanced usage

## ❓ Quick Troubleshooting

**Script won't run?**
```bash
chmod +x run_decision_maker_example.sh
./run_decision_maker_example.sh
```

**Missing dependencies?**
```bash
pip install pyyaml openai anthropic sentence-transformers aiohttp tenacity
```

**API errors?**
- Check your `.env` file has valid API keys
- Ensure keys start with correct prefixes (sk-, sk-ant-, xai-)

---

**Need help?** See the full [README.md](README.md) for detailed documentation.
