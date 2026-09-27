"""Serve v6 reader UI with the unchanged, already verified prediction engine."""
from pathlib import Path
import serve_prediction_demo as original

if __name__=='__main__':
    original.OUTPUT=Path(__file__).resolve().parents[1]/'artifacts/final_presentation/end_to_end_v6_readable'
    original.main()
