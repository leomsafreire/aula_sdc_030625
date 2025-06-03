# Soccer Action Value Analysis

This project analyzes soccer actions and player value using StatsBomb data and the `socceraction` library (which is amazing and you should also check its own repo). It includes a pipeline of four scripts to process and value actions, followed by Jupyter notebooks for further analysis.

## Project Structure

- **Scripts:**
  1. `1-load-and-convert-statsbomb-data.py`
  2. `2-compute-features-and-labels.py`
  3. `3-estimate-scoring-and-conceding-probabilities.py`
  4. `4-compute-vaep-values.py`
- **Notebooks:**
  - `player_passing_clusters.ipynb`
  - `pitch_zone_value_analysis.ipynb`

## Setup Instructions

1. **Clone the repository** and navigate to the project folder:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3.10 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

**You must run the four scripts in order before using the notebooks.**

```bash
python 1-load-and-convert-statsbomb-data.py
python 2-compute-features-and-labels.py
python 3-estimate-scoring-and-conceding-probabilities.py
python 4-compute-vaep-values.py
```

Each script will generate intermediate data in the `data/` folder.

## Using the Notebooks

After running the scripts, you can open and explore the notebooks:

```bash
jupyter notebook
```

## Environment

This project was developed with Python 3.10 and the `socceraction` library. For best results, use the provided `requirements.txt` and Python 3.10.

## License

MIT License 