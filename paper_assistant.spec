
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('frontend/build', 'frontend/build'),
        ('backend', 'backend'),
        ('backend/.env', '.'),
    ],
    hiddenimports=[
        'flask',
        'openai',
        'requests',
        'flask_cors',
        'python_decouple',
        'habanero',
        'semanticscholar',
        'httpx',
        'flask_limiter',
        'werkzeug.serving',
        'werkzeug.middleware',
        'flask.json.provider',
        'json',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AI论文写作助手',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='frontend/public/favicon.ico'
)
        