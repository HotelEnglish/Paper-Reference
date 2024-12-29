# reference.py - 处理文献搜索相关功能
from flask import Flask, jsonify
from flask_cors import CORS
from decouple import config
import requests
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置速率限制
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# 配置CORS
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=CORS_ORIGINS)

def search_semantic_scholar(query: str, max_results: int = 5):
    """使用Semantic Scholar API搜索文献"""
    try:
        logger.info(f"Searching Semantic Scholar for: {query}")
        url = f"https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,journal,venue,publicationVenue,citationCount"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            papers = []
            
            for paper in data.get('data', []):
                paper_info = {
                    'title': paper.get('title', ''),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'year': paper.get('year'),
                    'journal': (paper.get('publicationVenue', {}) or {}).get('name', ''),
                    'source': 'Semantic Scholar'
                }
                papers.append(paper_info)
            
            return papers
            
    except Exception as e:
        logger.error(f"Semantic Scholar search error: {str(e)}")
        return []

def format_apa7_citation(paper):
    """将论文信息转换为APA第7版格式"""
    authors = paper.get('authors', [])
    if len(authors) > 1:
        author_text = f"{', '.join(authors[:-1])}, & {authors[-1]}"
    else:
        author_text = authors[0] if authors else "Unknown"
    
    year = paper.get('year', 'n.d.')
    title = paper.get('title', '')
    journal = paper.get('journal', '')
    
    citation = f"{author_text}. ({year}). {title}. "
    if journal:
        citation += f"{journal}. "
    citation += f" [Semantic Scholar]"
    
    return citation

@app.route('/api/search-references', methods=['POST'])
@limiter.limit("10 per minute")
def search_references():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'success': False, 'error': '请输入搜索关键词'}), 400
            
        papers = search_semantic_scholar(query)
        
        for paper in papers:
            paper['citation'] = format_apa7_citation(paper)
            
        return jsonify({
            'success': True,
            'papers': papers
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
