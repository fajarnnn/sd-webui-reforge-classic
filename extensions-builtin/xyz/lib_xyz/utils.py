import csv
from io import StringIO
from itertools import chain

from modules import sd_vae


def boolean_choice(reverse: bool = False):
    def choice():
        return ["False", "True"] if reverse else ["True", "False"]

    return choice


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x


def list_to_csv_string(data_list):
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()


def csv_string_to_list_strip(data_str):
    return list(
        map(
            str.strip,
            chain.from_iterable(csv.reader(StringIO(data_str), skipinitialspace=True)),
        )
    )


def find_vae(name: str):
    if name.lower() in ["auto", "automatic"]:
        return sd_vae.unspecified
    if name.lower() == "none":
        return None
    else:
        choices = [x for x in sorted(sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
        if len(choices) == 0:
            print(f"No VAE found for {name}; using automatic")
            return sd_vae.unspecified
        else:
            return sd_vae.vae_dict[choices[0]]
