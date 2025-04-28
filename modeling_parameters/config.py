import os
import glob


PRIM_PARTICLE = 'p'
THETA = 30


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
READ_DIR = os.path.join(ROOT_DIR, 'modeling_parameters', 'read')

READ_CORSIKA_DIR = os.path.join(READ_DIR, 'corsika')
READ_GEANT_DIR = os.path.join(READ_DIR, 'geant')

READ_CORSIKA_OUTPUT_DIR = os.path.join(READ_CORSIKA_DIR, 'output', f'{PRIM_PARTICLE}{THETA}')
READ_GEANT_OUTPUT_DIR = os.path.join(READ_GEANT_DIR, 'output', f'{PRIM_PARTICLE}{THETA}')


RECONSTRUCTION_DIR = os.path.join(ROOT_DIR, 'modeling_parameters', 'reconstruction')

RECONSTRUCTION_CORSIKA_DIR = os.path.join(RECONSTRUCTION_DIR , 'corsika')
RECONSTRUCTION_GEANT_DIR = os.path.join(RECONSTRUCTION_DIR , 'geant')

RECONSTRUCTION_CORSIKA_OUTPUT_DIR = os.path.join(RECONSTRUCTION_CORSIKA_DIR, 'output', f'{PRIM_PARTICLE}{THETA}')
RECONSTRUCTION_GEANT_OUTPUT_DIR = os.path.join(RECONSTRUCTION_GEANT_DIR, 'output', f'{PRIM_PARTICLE}{THETA}')

COORDINATES_PATH = os.path.join(DATA_DIR, 'coordinates.csv')
