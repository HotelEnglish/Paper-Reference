import os
import subprocess
import shutil
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_node_installation():
    """Verify Node.js is installed"""
    try:
        subprocess.run(['node', '--version'], check=True, capture_output=True)
        subprocess.run(['npm', '--version'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("Node.js and npm are required but not found")
    except FileNotFoundError:
        raise RuntimeError("Node.js and npm are required but not found")

def build_frontend():
    """Build frontend with detailed error handling"""
    logger.info("Starting frontend build process...")
    
    frontend_dir = Path('frontend')
    build_dir = frontend_dir / 'build'
    
    # Clean existing build
    if build_dir.exists():
        logger.info("Cleaning existing build directory...")
        shutil.rmtree(build_dir)
    
    try:
        # Change to frontend directory
        os.chdir(frontend_dir)
        
        # Install dependencies
        logger.info("Installing frontend dependencies...")
        subprocess.run('npm install', shell=True, check=True, 
                      stderr=subprocess.PIPE)
        
        # Build frontend
        logger.info("Building frontend...")
        subprocess.run('npm run build', shell=True, check=True,
                      stderr=subprocess.PIPE)
        
        # Return to root directory
        os.chdir('..')
        
        # Verify build
        if not build_dir.exists():
            raise RuntimeError("Build directory not created")
        
        required_files = ['index.html', 'static']
        for file in required_files:
            if not (build_dir / file).exists():
                raise RuntimeError(f"Required file/directory not found: {file}")
        
        logger.info("Frontend build completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Build process failed: {e.stderr.decode()}")
        raise RuntimeError("Frontend build failed")
    except Exception as e:
        logger.error(f"Unexpected error during frontend build: {str(e)}")
        raise

def prepare_env():
    """Prepare environment files"""
    logger.info("Preparing environment files...")
    
    env_example = Path('backend/.env.example')
    env_file = Path('backend/.env')
    
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            logger.info("Created .env file from example")
        else:
            raise RuntimeError(".env.example file not found")

def build_executable():
    """Build the executable with comprehensive error handling"""
    try:
        logger.info("Starting build process...")
        
        # Verify Node.js installation
        verify_node_installation()
        
        # Prepare environment
        prepare_env()
        
        # Build frontend
        build_frontend()
        
        logger.info("Creating PyInstaller spec file...")
        spec_content = '''# -*- mode: python ; coding: utf-8 -*-

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
'''
        
        with open('paper_assistant.spec', 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        logger.info("Building executable with PyInstaller...")
        subprocess.run('pyinstaller --clean paper_assistant.spec', 
                      shell=True, check=True)
        
        logger.info("Build completed successfully!")
        logger.info("Executable can be found in the 'dist' directory")
        
    except Exception as e:
        logger.error(f"Build failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    build_executable() 