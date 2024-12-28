import React, { useState, useEffect } from 'react';
import { 
  Container, TextField, Button, Paper, Typography, Box, 
  Alert, Snackbar, CircularProgress, FormControl, 
  InputLabel, Select, MenuItem, IconButton, Tooltip,
  Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import axios from 'axios';

function App() {
  const [topic, setTopic] = useState('');
  const [paperContent, setPaperContent] = useState('');
  const [references, setReferences] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchResults, setSearchResults] = useState({});
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('gpt-4o-mini');
  const [copySuccess, setCopySuccess] = useState('');
  const [welcomeOpen, setWelcomeOpen] = useState(true);

  useEffect(() => {
    // 获取可用的模型列表
    const fetchModels = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/models');
        if (response.data.success) {
          setModels(response.data.models);
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };

    fetchModels();
  }, []);

  const handleSubmit = async () => {
    if (!topic.trim()) {
      setError('请输入论文主题');
      return;
    }

    setLoading(true);
    setError(null);
    setPaperContent('');
    setReferences([]);
    setSearchResults({});
    
    try {
      const response = await axios.post('http://localhost:5000/api/generate-paper', {
        text: topic.trim(),
        language: 'zh',
        format: 'apa7',
        model: selectedModel
      });
      
      if (response.data.success) {
        setPaperContent(response.data.paper);
        setReferences(response.data.references);
        setSearchResults(response.data.search_results);
      } else {
        throw new Error(response.data.error || '生成失败，请稍后重试');
      }
    } catch (error) {
      console.error('Error details:', error);
      let errorMessage = '服务暂时不可用，请稍后重试';
      
      if (error.response) {
        console.error('Server error response:', error.response.data);
        errorMessage = error.response.data.error || errorMessage;
      } else if (error.request) {
        console.error('No response received:', error.request);
        errorMessage = '无法连接到服务器，请检查网络连接';
      } else {
        console.error('Request error:', error.message);
        errorMessage = error.message || errorMessage;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // 复制功能
  const handleCopy = async (text, type) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopySuccess(`${type}已复制到剪贴板`);
    } catch (err) {
      setCopySuccess('复制失败，请手动复制');
    }
  };

  // 欢迎对话框内容
  const WelcomeDialog = () => (
    <Dialog
      open={welcomeOpen}
      onClose={() => setWelcomeOpen(false)}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>
        欢迎使用 AI 论文写作助手
      </DialogTitle>
      <DialogContent>
        <Typography paragraph>
          这是一个强大的学术论文写作工具，能够帮助您：
        </Typography>
        <Typography component="div">
          <ul>
            <li>自动搜索相关学术文献</li>
            <li>生成包含引用的论文内容</li>
            <li>支持多个AI模型选择</li>
            <li>自动生成APA格式引用</li>
            <li>一键复制论文内容和引用</li>
          </ul>
        </Typography>
        <Typography paragraph>
          使用步骤：
        </Typography>
        <Typography component="div">
          <ol>
            <li>选择您想使用的AI模型</li>
            <li>输入论文主题或研究方向</li>
            <li>点击"生成论文"按钮</li>
            <li>等待系统生成内容</li>
            <li>使用复制功能获取内容</li>
          </ol>
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setWelcomeOpen(false)} color="primary">
          开始使用
        </Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <Container maxWidth="md">
      <WelcomeDialog />
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          AI论文写作助手
        </Typography>
        
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
            请输入论文主题，系统将自动生成包含引用的学术论文内容
          </Typography>
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>选择AI模型</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              label="选择AI模型"
              disabled={loading}
            >
              {models.map((model) => (
                <MenuItem key={model.id} value={model.id}>
                  {model.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <TextField
            fullWidth
            multiline
            rows={3}
            variant="outlined"
            label="输入论文主题"
            placeholder="例如：人工智能在教育领域的应用"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            sx={{ mb: 2 }}
            disabled={loading}
          />
          
          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={loading}
            sx={{ mb: 2 }}
            fullWidth
          >
            {loading ? (
              <>
                <CircularProgress size={24} sx={{ mr: 1 }} color="inherit" />
                正在生成论文内容...
              </>
            ) : '生成论文'}
          </Button>
        </Paper>

        {paperContent && (
          <Paper sx={{ p: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                生成的论文内容：
              </Typography>
              <Tooltip title="复制论文内容">
                <IconButton onClick={() => handleCopy(paperContent, '论文内容')}>
                  <ContentCopyIcon />
                </IconButton>
              </Tooltip>
            </Box>
            <Box sx={{ 
              p: 2, 
              bgcolor: '#f5f5f5', 
              borderRadius: 1,
              whiteSpace: 'pre-wrap',
              fontFamily: 'serif',
              lineHeight: 1.8
            }}>
              <Typography variant="body1">
                {paperContent}
              </Typography>
            </Box>
          </Paper>
        )}

        {references.length > 0 && (
          <Paper sx={{ p: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                参考文献：
              </Typography>
              <Tooltip title="复制所有参考文献">
                <IconButton onClick={() => handleCopy(references.join('\n'), '参考文献')}>
                  <ContentCopyIcon />
                </IconButton>
              </Tooltip>
            </Box>
            {references.map((ref, index) => (
              <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography 
                  variant="body2" 
                  sx={{ 
                    pl: 3, 
                    textIndent: '-1.5em',
                    flex: 1,
                    fontFamily: 'serif'
                  }}
                >
                  {`${index + 1}. ${ref}`}
                </Typography>
                <Tooltip title="复制此条引用">
                  <IconButton 
                    size="small"
                    onClick={() => handleCopy(ref, '引用')}
                  >
                    <ContentCopyIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            ))}
          </Paper>
        )}

        {Object.keys(searchResults).length > 0 && (
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              文献搜索结果：
            </Typography>
            {Object.entries(searchResults).map(([source, papers]) => (
              papers.length > 0 && (
                <Box key={source} sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" color="primary" gutterBottom>
                    {source === 'crossref' ? 'Crossref' :
                     source === 'semantic_scholar' ? 'Semantic Scholar' :
                     'Google Scholar'} 搜索结果：
                  </Typography>
                  {papers.map((paper, index) => (
                    <Box 
                      key={index}
                      sx={{ 
                        p: 1.5,
                        mb: 1,
                        borderLeft: '3px solid #1976d2',
                        bgcolor: '#f8f8f8'
                      }}
                    >
                      <Typography variant="body2" gutterBottom>
                        <strong>标题：</strong> {paper.title}
                      </Typography>
                      <Typography variant="body2">
                        <strong>引用格式：</strong> {paper.citation}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              )
            ))}
          </Paper>
        )}

        <Snackbar 
          open={!!error} 
          autoHideDuration={6000} 
          onClose={() => setError(null)}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        >
          <Alert 
            severity="error" 
            onClose={() => setError(null)}
            variant="filled"
          >
            {error}
          </Alert>
        </Snackbar>

        <Snackbar 
          open={!!copySuccess} 
          autoHideDuration={2000} 
          onClose={() => setCopySuccess('')}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert 
            severity="success" 
            onClose={() => setCopySuccess('')}
            variant="filled"
          >
            {copySuccess}
          </Alert>
        </Snackbar>
      </Box>
    </Container>
  );
}

export default App;
