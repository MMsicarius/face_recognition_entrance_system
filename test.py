import cv2

from simple_facerec import SimpleFacerec

x = "Michael Maitland"
# Encode faces from a folder
sfr = SimpleFacerec()
y = sfr.search_credentials(x)

