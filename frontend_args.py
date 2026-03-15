from options import args_parser


# 将前端传入的参数替换初始值
def override_args(frontend_params):
    args = args_parser()

    for key, value in frontend_params.items():
        if hasattr(args, key):
            old_value = getattr(args, key)
            try:
                # 保持原参数类型
                if isinstance(old_value, bool):
                    # 特殊处理布尔值
                    new_value = value.lower() in ['1', 'true', 'yes']
                else:
                    new_value = type(old_value)(value)
                setattr(args, key, new_value)
            except Exception as e:
                print(f"[警告] 参数 {key} 的值 '{value}' 转换失败，保留默认值 {old_value}。错误：{e}")
        else:
            print(f"[提示] 参数 {key} 不在 args 中，跳过")

    # 处理差分隐私策略
    if 'dp_level' in frontend_params:
        dp_level = frontend_params['dp_level'].lower()
        if dp_level == 'high':
            args.client_add_noise = 0
            args.edge_add_noise = 0
        else:
            args.client_add_noise = 1
            args.edge_add_noise = 1

    if args.dataset.lower() in ['cifar10']:
        args.input_channels = 3
    elif args.dataset.lower() in ['mnist', 'fmnist', 'fashion-mnist']:
        args.input_channels = 1

    return args
