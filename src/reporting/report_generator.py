import csv
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_timestamp_str(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_framewise_csv(self, video_name, frame_data):
        """
        Generates a CSV file containing frame-by-frame counts.
        frame_data format: list of dicts: [{'frame': 1, 'timestamp_sec': 0.1, 'Elephants': 2, 'Zebras': 0}, ...]
        """
        filename = f"{video_name}_framewise_{self._get_timestamp_str()}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        if not frame_data:
            logger.warning("No frame data provided for CSV generation.")
            return filepath
            
        # Get all unique keys (columns) from the list of dicts
        keys = set()
        for fd in frame_data:
            keys.update(fd.keys())
            
        # Ensure 'frame' and 'timestamp_sec' are first
        fieldnames = ['frame', 'timestamp_sec']
        other_keys = sorted(list(keys - set(fieldnames)))
        fieldnames.extend(other_keys)

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(frame_data)
            logger.info(f"Successfully generated frame-wise CSV: {filepath}")
        except Exception as e:
            logger.error(f"Error generating frame-wise CSV: {e}")
            
        return filepath

    def generate_video_summary(self, video_name, unique_ids_dict, appearance_times=None):
        """
        Generates both CSV and JSON summaries per video.
        unique_ids_dict format: {'Elephant': {1, 2}, 'Zebra': {3}}
        appearance_times format: {'Elephant': {1: 0.5, 2: 10.2}, 'Zebra': {3: 4.1}}
        """
        timestamp = self._get_timestamp_str()
        csv_filename = f"{video_name}_summary_{timestamp}.csv"
        json_filename = f"{video_name}_summary_{timestamp}.json"
        
        csv_filepath = os.path.join(self.output_dir, csv_filename)
        json_filepath = os.path.join(self.output_dir, json_filename)
        
        total_unique = 0
        species_counts = {}
        detailed_records = []
        
        for species, ids in unique_ids_dict.items():
            count = len(ids)
            species_counts[species] = count
            total_unique += count
            
            for obj_id in ids:
                first_seen = "Unknown"
                if appearance_times and species in appearance_times and obj_id in appearance_times[species]:
                     first_seen = appearance_times[species][obj_id]
                     
                detailed_records.append({
                    'species': species,
                    'track_id': obj_id,
                    'first_appearance_sec': first_seen
                })

        summary_data = {
            'video_name': video_name,
            'total_unique_animals': total_unique,
            'species_wise_count': species_counts,
            'detailed_records': detailed_records
        }

        # Write JSON
        try:
            with open(json_filepath, 'w') as f:
                json.dump(summary_data, f, indent=4)
            logger.info(f"Successfully generated video summary JSON: {json_filepath}")
        except Exception as e:
            logger.error(f"Error generating video summary JSON: {e}")

        # Write CSV
        try:
            with open(csv_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Video Name', video_name])
                writer.writerow(['Total Unique Animals', total_unique])
                writer.writerow([])
                writer.writerow(['Species', 'Count'])
                for species, count in species_counts.items():
                    writer.writerow([species, count])
                writer.writerow([])
                
                # Write detailed appearances if available
                if appearance_times:
                    writer.writerow(['Species', 'Track ID', 'First Appearance (Sec)'])
                    for p in detailed_records:
                         writer.writerow([p['species'], p['track_id'], p['first_appearance_sec']])
                         
            logger.info(f"Successfully generated video summary CSV: {csv_filepath}")
        except Exception as e:
            logger.error(f"Error generating video summary CSV: {e}")

        return csv_filepath, json_filepath
