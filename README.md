# updated_cvat


1) обновить файлы base.in и bace.txt в cvat/requirements
2) обновить файл registry.py в cvat/apps/dataset_manager/formats/registry.py
3) добавить файл yolo5.py и папку yolo_formater в cvat/apps/dataset_manager
4) перезапуск приложения docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build