import os


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)

    return f"{opt.label}: {x}"


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def format_remove_path(p, opt, x):
    return os.path.basename(x)


def format_nothing(p, opt, x):
    return ""
