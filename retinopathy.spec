# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['retinopathy.py'],
             pathex=['/Users/cashconverters/midlands/reti'],
             binaries=[],
             datas=[('model.h5', '.')],
             hiddenimports=['h5py', 'h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy', 'pywt._extensions._cwt', 'tensorflow_core.python', 'pywrap_tensorflow'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='retinopathy',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
