
from deeplens.optics import Lensgroup

# Load design file
lens = Lensgroup(filename='./final_lens.json')
# add supplementary properties
lens.wave = [520]
lens.is_asphere = False
lens.is_conic = False
lens.is_sphere = True
# change fov
lens.hfov = 0.20

# # Export PNG of current layout
# lens.analysis(draw_layout=True)

# Perform diameter pruning
lens.prune(outer=0.2)

# Export PNG of corrected layout
lens.analysis(draw_layout=True)