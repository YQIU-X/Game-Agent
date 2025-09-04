// 测试权重文件API
fetch('http://localhost:3000/api/models')
  .then(response => response.json())
  .then(data => {
    console.log('权重文件API响应:', data);
    if (data.success && data.models.length > 0) {
      console.log('找到权重文件:', data.models.length, '个');
      console.log('第一个文件:', data.models[0]);
    } else {
      console.log('没有找到权重文件');
    }
  })
  .catch(error => {
    console.error('API请求失败:', error);
  });

