def snake_to_pascal(snake_str):
    components = snake_str.split('_')
    pascal_case_str = ''.join(x.title() for x in components)
    return pascal_case_str