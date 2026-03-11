import pytest
import os
import sys
import json
import csv
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.reporting.report_generator import ReportGenerator

@pytest.fixture
def test_dir():
    """Create a temporary test directory and clean it up after"""
    dir_name = "test_outputs"
    os.makedirs(dir_name, exist_ok=True)
    yield dir_name
    shutil.rmtree(dir_name)

def test_actual_report_generation(test_dir):
    """Test generating real CSV and JSON files using the ReportGenerator."""
    
    reporter = ReportGenerator(output_dir=test_dir)
    video_name = "test_safari_video"
    
    # Simulate tracking 2 Elephants and 1 Zebra
    unique_ids_dict = {
        'Elephant': {1, 2},
        'Zebra': {3}
    }
    
    # Simulate timestamps of when they first appeared
    appearance_times = {
        'Elephant': {1: 1.5, 2: 12.0},
        'Zebra': {3: 4.2}
    }
    
    # Simulate frame-by-frame populations
    frame_data = [
        {'frame': 1, 'timestamp_sec': 0.05, 'Elephant': 0, 'Zebra': 0},
        {'frame': 30, 'timestamp_sec': 1.5, 'Elephant': 1, 'Zebra': 0},  # Id 1 appears
        {'frame': 84, 'timestamp_sec': 4.2, 'Elephant': 1, 'Zebra': 1},  # Id 3 appears
        {'frame': 240, 'timestamp_sec': 12.0, 'Elephant': 2, 'Zebra': 1} # Id 2 appears
    ]
    
    # Generate the actual files
    framewise_csv = reporter.generate_framewise_csv(video_name, frame_data)
    summary_csv, summary_json = reporter.generate_video_summary(video_name, unique_ids_dict, appearance_times)
    
    # Verify Framewise CSV
    assert os.path.exists(framewise_csv), "Framewise CSV was not created"
    with open(framewise_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 4
        assert rows[1]['frame'] == '30'
        assert rows[1]['Elephant'] == '1'
        
    # Verify Summary JSON
    assert os.path.exists(summary_json), "Summary JSON was not created"
    with open(summary_json, 'r') as f:
        data = json.load(f)
        assert data['total_unique_animals'] == 3
        assert data['species_wise_count']['Elephant'] == 2
        assert data['species_wise_count']['Zebra'] == 1
        assert data['video_name'] == "test_safari_video"
        
        records = data['detailed_records']
        # Find the record for Zebra ID 3
        zebra_record = next(r for r in records if r['species'] == 'Zebra' and r['track_id'] == 3)
        assert zebra_record['first_appearance_sec'] == 4.2
        
    # Verify Summary CSV
    assert os.path.exists(summary_csv), "Summary CSV was not created"
    with open(summary_csv, 'r') as f:
        text = f.read()
        assert "test_safari_video" in text
        assert "Total Unique Animals,3" in text
        assert "Elephant,2" in text

