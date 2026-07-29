import base64
import os
from PIL import Image, ImageEnhance

def process_image(img_path):
    img = Image.open(img_path).convert("RGB")
    
    # Crop to center square
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    img = img.crop((left, top, left + size, top + size))
    
    # Resize for SVG embedding
    img = img.resize((280, 280), Image.Resampling.LANCZOS)
    
    # Enhance contrast and lower saturation slightly for the tech look
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(0.7)
    
    temp_path = "assets/portrait_hud_temp.jpg"
    img.save(temp_path, format="JPEG", quality=85)
    
    with open(temp_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_svg():
    b64_img = process_image("assets/portrait.jpg")
    
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600" viewBox="0 0 1200 600">
    <defs>
        <style>
            .bg {{ fill: #050A15; }}
            .text-main {{ font-family: 'Courier New', 'Consolas', monospace; fill: #00F0FF; }}
            .text-muted {{ font-family: 'Courier New', 'Consolas', monospace; fill: #0088AA; }}
            .text-gold {{ font-family: 'Courier New', 'Consolas', monospace; fill: #FFB300; font-weight: bold; }}
            
            .ring-fast {{ transform-origin: 300px 300px; animation: spin-fast 12s linear infinite; }}
            .ring-slow-rev {{ transform-origin: 300px 300px; animation: spin-rev 24s linear infinite; }}
            .ring-medium {{ transform-origin: 300px 300px; animation: spin-fast 18s linear infinite; }}
            .pulse {{ animation: pulsing 2s ease-in-out infinite alternate; }}
            
            /* CSS Glow (using drop-shadow is cleaner than SVG filters for simple glows) */
            .glow {{ filter: drop-shadow(0 0 6px #00F0FF); }}
            .glow-gold {{ filter: drop-shadow(0 0 6px #FFB300); }}
            
            @keyframes spin-fast {{ 100% {{ transform: rotate(360deg); }} }}
            @keyframes spin-rev {{ 100% {{ transform: rotate(-360deg); }} }}
            @keyframes pulsing {{ 0% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
        </style>
        <clipPath id="avatar-clip">
            <circle cx="300" cy="300" r="140" />
        </clipPath>
        
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#003344" stroke-width="0.5" opacity="0.3"/>
        </pattern>
        
        <pattern id="hex-bg" width="34.6" height="60" patternUnits="userSpaceOnUse" patternTransform="scale(0.5)">
             <path d="M17.3 0L34.6 10L34.6 30L17.3 40L0 30L0 10Z M17.3 60L34.6 50L34.6 30L17.3 40L0 30L0 50Z" fill="none" stroke="#003344" stroke-width="1" opacity="0.1"/>
        </pattern>
    </defs>
    
    <!-- Background layers -->
    <rect width="1200" height="600" class="bg" />
    <rect width="1200" height="600" fill="url(#hex-bg)" />
    
    <!-- Scanline Effect -->
    <rect width="1200" height="5" fill="#00F0FF" opacity="0.05">
        <animate attributeName="y" values="-10;610" dur="4s" repeatCount="indefinite" />
    </rect>
    
    <!-- Left: Arc Reactor Profile -->
    <g class="glow">
        <!-- Outer target ring -->
        <circle cx="300" cy="300" r="190" fill="none" stroke="#003344" stroke-width="2" />
        <circle cx="300" cy="300" r="190" fill="none" stroke="#00F0FF" stroke-width="3" stroke-dasharray="20 40 60 40" class="ring-slow-rev" />
        
        <!-- Medium dashed ring -->
        <circle cx="300" cy="300" r="170" fill="none" stroke="#00A3FF" stroke-width="8" stroke-dasharray="10 15 30 15" class="ring-fast" />
        <circle cx="300" cy="300" r="160" fill="none" stroke="#00F0FF" stroke-width="2" stroke-dasharray="4 8" class="ring-medium" />
        
        <!-- Inner framing ring -->
        <circle cx="300" cy="300" r="148" fill="none" stroke="#00F0FF" stroke-width="4" stroke-dasharray="150 20" class="ring-slow-rev" />
        <circle cx="300" cy="300" r="142" fill="none" stroke="#FFB300" stroke-width="1" stroke-dasharray="2 4" />
        
        <!-- Crosshairs -->
        <line x1="80" y1="300" x2="105" y2="300" stroke="#00F0FF" stroke-width="2" />
        <line x1="495" y1="300" x2="520" y2="300" stroke="#00F0FF" stroke-width="2" />
        <line x1="300" y1="80" x2="300" y2="105" stroke="#00F0FF" stroke-width="2" />
        <line x1="300" y1="495" x2="300" y2="520" stroke="#00F0FF" stroke-width="2" />
    </g>
    
    <!-- Profile Image -->
    <image x="160" y="160" width="280" height="280" clip-path="url(#avatar-clip)" preserveAspectRatio="xMidYMid slice" href="data:image/jpeg;base64,{b64_img}" />
    <!-- Cyan HUD overlay on image -->
    <circle cx="300" cy="300" r="140" fill="#00F0FF" opacity="0.15" class="pulse"/>
    
    <!-- Center separator line -->
    <line x1="560" y1="100" x2="560" y2="500" stroke="#003344" stroke-width="2" />
    <circle cx="560" cy="300" r="4" fill="#00F0FF" class="pulse" />
    
    <!-- Right: Data Readout Panel -->
    <g transform="translate(620, 120)">
        <text x="0" y="0" font-size="28" font-weight="bold" class="text-main glow">[ RON_OS // SYSTEM ONLINE ]</text>
        <line x1="0" y1="20" x2="480" y2="20" stroke="#0088AA" stroke-width="1" />
        
        <!-- Animated typing effect cursor -->
        <rect x="440" y="-22" width="15" height="24" fill="#00F0FF">
            <animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite" />
        </rect>
        
        <text x="0" y="60" font-size="16" class="text-muted">SUBJECT_ID   :</text>
        <text x="160" y="60" font-size="18" font-weight="bold" class="text-main glow">VIVEK V RON</text>
        
        <text x="0" y="90" font-size="16" class="text-muted">DESIGNATION  :</text>
        <text x="160" y="90" font-size="18" class="text-gold glow-gold">MACHINE LEARNING ENGINEER</text>
        
        <text x="0" y="120" font-size="16" class="text-muted">LOCATION     :</text>
        <text x="160" y="120" font-size="16" class="text-main">HUBBALLI, KA, IN</text>
        
        <text x="0" y="150" font-size="16" class="text-muted">STATUS       :</text>
        <text x="160" y="150" font-size="16" class="text-main pulse">BUILDING INTELLIGENT WEB APPS</text>
        
        <line x1="0" y1="180" x2="480" y2="180" stroke="#003344" stroke-width="1" />
        
        <!-- Skills Bars -->
        <text x="0" y="220" font-size="18" font-weight="bold" class="text-main glow">&gt;&gt; CORE.TECHNOLOGIES</text>
        
        <!-- Python -->
        <text x="0" y="260" font-size="16" class="text-muted">PYTHON</text>
        <rect x="140" y="248" width="280" height="12" fill="#003344" />
        <rect x="140" y="248" width="266" height="12" fill="#00F0FF" class="glow pulse" />
        <text x="435" y="260" font-size="14" class="text-main">95%</text>
        
        <!-- Java -->
        <text x="0" y="300" font-size="16" class="text-muted">JAVA</text>
        <rect x="140" y="288" width="280" height="12" fill="#003344" />
        <rect x="140" y="288" width="238" height="12" fill="#00A3FF" />
        <text x="435" y="300" font-size="14" class="text-main">85%</text>
        
        <!-- Google Cloud -->
        <text x="0" y="340" font-size="16" class="text-muted">GCP / INFRA</text>
        <rect x="140" y="328" width="280" height="12" fill="#003344" />
        <rect x="140" y="328" width="224" height="12" fill="#FFB300" class="glow-gold pulse" />
        <text x="435" y="340" font-size="14" class="text-gold">80%</text>
        
        <!-- Connections -->
        <line x1="0" y1="380" x2="480" y2="380" stroke="#003344" stroke-width="1" />
    </g>
    
    <!-- Hexagon accents top right -->
    <g transform="translate(1100, 40)" fill="none" stroke="#00F0FF" stroke-width="1.5" opacity="0.6">
        <polygon points="20,0 40,10 40,30 20,40 0,30 0,10" class="glow"/>
    </g>
    <g transform="translate(1080, 75)" fill="none" stroke="#0088AA" stroke-width="1" opacity="0.4">
        <polygon points="20,0 40,10 40,30 20,40 0,30 0,10" />
    </g>
    <g transform="translate(1120, 75)" fill="none" stroke="#FFB300" stroke-width="1" opacity="0.5" class="pulse">
        <polygon points="20,0 40,10 40,30 20,40 0,30 0,10" class="glow-gold" />
    </g>
    <g transform="translate(1060, 40)" fill="none" stroke="#003344" stroke-width="1" opacity="0.5">
        <polygon points="20,0 40,10 40,30 20,40 0,30 0,10" />
    </g>
    
    <!-- Tech decorative nodes -->
    <circle cx="50" cy="50" r="15" fill="none" stroke="#00F0FF" stroke-width="2" class="glow" />
    <circle cx="50" cy="50" r="5" fill="#00F0FF" />
    <line x1="65" y1="50" x2="200" y2="50" stroke="#00F0FF" stroke-width="1" stroke-dasharray="4 2" />
    
    <circle cx="1150" cy="550" r="15" fill="none" stroke="#FFB300" stroke-width="2" class="glow-gold" />
    <circle cx="1150" cy="550" r="5" fill="#FFB300" />
    <line x1="1135" y1="550" x2="900" y2="550" stroke="#FFB300" stroke-width="1" stroke-dasharray="4 2" />
</svg>"""
    
    with open("output/hud_banner.svg", "w", encoding="utf-8") as f:
        f.write(svg)
        
if __name__ == "__main__":
    generate_svg()
    print("Generated output/hud_banner.svg successfully.")
