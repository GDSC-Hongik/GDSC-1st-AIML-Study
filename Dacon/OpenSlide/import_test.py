def import_slide() :
    OPENSLIDE_PATH = r'C:\openslide-win64-20221111\bin'

    import os
    if hasattr(os, 'add_dll_directory'):
        # Python >= 3.8 on Windows
        with os.add_dll_directory(OPENSLIDE_PATH):
            from openslide import open_slide
            print('import 성공')
    else:
        import openslide
        print('import 성공2')
