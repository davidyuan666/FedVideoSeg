# FedVideoQA Makefile

.PHONY: help install quick-start full-start prepare-data clean

help:
	@echo "FedVideoQA 联邦学习框架"
	@echo ""
	@echo "可用命令:"
	@echo "  install      - 安装依赖包"
	@echo "  prepare-data - 准备示例数据集"
	@echo "  quick-start  - 快速启动(演示模式)"
	@echo "  full-start   - 完整启动联邦学习"
	@echo "  clean        - 清理生成的文件"

install:
	pip install -r requirements.txt

prepare-data:
	python scripts/prepare_dataset.py --data_dir data/ --num_videos 15

quick-start:
	python quick_start.py

full-start:
	python federated_launcher.py --config config/federated_config.json

clean:
	rm -rf logs/ results/ data/dataset.json
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete


commit:
	git add .
	git commit -m "update"
	git push origin main