import xml.etree.ElementTree as ET
import sys

try:
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()
    
    # Namespaces
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    text = []
    for p in root.findall('.//w:p', ns):
        p_text = []
        for t in p.findall('.//w:t', ns):
            if t.text:
                p_text.append(t.text)
        if p_text:
            text.append(''.join(p_text))
            
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(text))
        
    print("Text extracted successfully.")
except Exception as e:
    print(f"Error: {e}")
