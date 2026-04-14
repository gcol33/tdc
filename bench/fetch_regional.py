"""Fetch just the NASA POWER regional dataset."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from prepare_real_data import fetch_nasa_power_regional
fetch_nasa_power_regional()
