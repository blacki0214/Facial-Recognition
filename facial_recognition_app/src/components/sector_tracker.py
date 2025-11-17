import cv2
import numpy as np
from collections import defaultdict, deque
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings

class SectorTracker:
    def __init__(self):
        self.sector_history = defaultdict(lambda: deque(maxlen=30))  # 30 frames history
        self.sector_names = settings.SECTOR_NAMES
        self.sector_colors = settings.SECTOR_COLORS
        
    def get_sector(self, x, w, frame_width):
        """Determine which sector the face is in"""
        face_center_x = x + w // 2
        sector_width = frame_width // 3
        
        if face_center_x < sector_width:
            return "Left"
        elif face_center_x < sector_width * 2:
            return "Center"
        else:
            return "Right"
    
    def update_sector(self, person_id, sector):
        """Update sector history for a person"""
        self.sector_history[person_id].append(sector)
    
    def get_dominant_sector(self, person_id):
        """Get the most frequent sector for a person"""
        if person_id not in self.sector_history or len(self.sector_history[person_id]) == 0:
            return None
        
        history = list(self.sector_history[person_id])
        return max(set(history), key=history.count)
    
    def draw_sectors(self, frame):
        """Draw sector divisions on frame"""
        height, width = frame.shape[:2]
        sector_width = width // 3
        
        # Draw vertical lines
        cv2.line(frame, (sector_width, 0), (sector_width, height), (255, 255, 255), 2)
        cv2.line(frame, (sector_width * 2, 0), (sector_width * 2, height), (255, 255, 255), 2)
        
        # Add sector labels
        for i, sector_name in enumerate(self.sector_names):
            x_pos = sector_width // 2 + i * sector_width
            cv2.putText(frame, sector_name, (x_pos - 40, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.sector_colors[sector_name], 2)
        
        return frame
    
    def get_sector_stats(self):
        """Get statistics of people in each sector"""
        stats = {sector: [] for sector in self.sector_names}
        
        for person_id, history in self.sector_history.items():
            if len(history) > 0:
                dominant_sector = self.get_dominant_sector(person_id)
                if dominant_sector:
                    stats[dominant_sector].append(person_id)
        
        return stats