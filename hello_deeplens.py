""" 
"Hello. world!" for DeepLens. 

In this code, we will load a lens from a file. Then we will plot the lens setup and render a sample image.
"""
from deeplens import Lensgroup

def main():
    lens = Lensgroup(filename='./lens_zoo/2P_Aspheric_FOV45_9_Wave450.json')
    lens.wave = [450, 450, 450]
    lens.analysis(draw_layout=True)

if __name__=='__main__':
    main()