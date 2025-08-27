#!/usr/bin/env python3
# 打印LIBERO任务名称

try:
    from libero.libero.benchmark import get_benchmark
    
    # 打印libero_spatial任务
    print("\n===== LIBERO_SPATIAL任务名称列表 =====")
    bm_spatial = get_benchmark('libero_spatial')()
    for i, task in enumerate(bm_spatial.get_task_names()):
        print(f"{i}: {task}")

    # 打印libero_goal任务
    print("\n===== LIBERO_GOAL任务名称列表 =====")
    bm_goal = get_benchmark('libero_goal')()
    for i, task in enumerate(bm_goal.get_task_names()):
        print(f"{i}: {task}")

    # 打印我们配置文件中使用的两个任务
    tasks = ["pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"]
    print("\n===== 配置文件中的任务 =====")
    for task in tasks:
        # 检查是否在spatial任务中
        if task in bm_spatial.get_task_names():
            idx = bm_spatial.get_task_names().index(task)
            print(f"在LIBERO_SPATIAL中：{task} (ID={idx})")
        # 检查是否在goal任务中
        elif task in bm_goal.get_task_names():
            idx = bm_goal.get_task_names().index(task)
            print(f"在LIBERO_GOAL中：{task} (ID={idx})")
        else:
            print(f"未找到任务: {task}")

except Exception as e:
    print(f"导入或执行失败: {e}") 