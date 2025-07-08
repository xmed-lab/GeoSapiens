dataset_info = dict(
    dataset_name='cbct_teeth',
    paper_info=dict(
        author='Marawan Elbatel',
        title='CBCT Teeth Landmark Detection',
        container='arXiv',
        year='2025',
        homepage='marwankefah.github.io',
    ),
    keypoint_info={
        0: dict(name='point1', id=0, color=[255, 0, 0], type='', swap=''),
        1: dict(name='point2', id=1, color=[255, 128, 0], type='', swap=''),
        2: dict(name='point3', id=2, color=[255, 255, 0], type='', swap=''),
        3: dict(name='point4', id=3, color=[128, 255, 0], type='', swap=''),
        4: dict(name='point5', id=4, color=[0, 255, 0], type='', swap=''),
        5: dict(name='point6', id=5, color=[0, 255, 128], type='', swap=''),
        6: dict(name='point7', id=6, color=[0, 255, 255], type='', swap=''),
        7: dict(name='point8', id=7, color=[0, 128, 255], type='', swap=''),
        8: dict(name='point9', id=8, color=[0, 0, 255], type='', swap=''),
        9: dict(name='point10', id=9, color=[128, 0, 255], type='', swap=''),
        10: dict(name='point11', id=10, color=[255, 0, 255], type='', swap=''),
        11: dict(name='point12', id=11, color=[255, 0, 128], type='', swap=''),
        12: dict(name='point13', id=12, color=[255, 0, 64], type='', swap=''),
        13: dict(name='point14', id=13, color=[255, 64, 0], type='', swap=''),
        14: dict(name='point15', id=14, color=[255, 128, 64], type='', swap=''),
        15: dict(name='point16', id=15, color=[128, 128, 128], type='', swap=''),
    },
    skeleton_info={},  # Optional: Define relationships if needed
    joint_weights=[1.] * 16,  # Equal weight for each point, adjust if needed
    sigmas=[0.01]*16  # Optional: Add if needed for evaluation
)
