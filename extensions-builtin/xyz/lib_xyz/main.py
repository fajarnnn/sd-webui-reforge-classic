import re
from copy import copy

from PIL import Image

from modules import images
from modules.processing import Processed
from modules.shared import state


def draw_xyz_grid(
    p,
    xs,
    ys,
    zs,
    x_labels,
    y_labels,
    z_labels,
    cell,
    draw_legend,
    include_lone_images,
    include_sub_grids,
    first_axes_processed,
    second_axes_processed,
    margin_size,
):
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    title_texts = [[images.GridAnnotation(z)] for z in z_labels]

    list_size = len(xs) * len(ys) * len(zs)

    processed_result = None

    state.job_count = list_size * p.n_iter

    def process_cell(x, y, z, ix, iy, iz):
        nonlocal processed_result

        def index(ix, iy, iz):
            return ix + iy * len(xs) + iz * len(xs) * len(ys)

        state.job = f"{index(ix, iy, iz) + 1} out of {list_size}"

        processed: Processed = cell(x, y, z, ix, iy, iz)

        if processed_result is None:
            # Use our first processed result object as a template container to hold our full results
            processed_result = copy(processed)
            processed_result.images = [None] * list_size
            processed_result.all_prompts = [None] * list_size
            processed_result.all_seeds = [None] * list_size
            processed_result.infotexts = [None] * list_size
            processed_result.index_of_first_image = 1

        idx = index(ix, iy, iz)
        if processed.images:
            # Non-empty list indicates some degree of success.
            processed_result.images[idx] = processed.images[0]
            processed_result.all_prompts[idx] = processed.prompt
            processed_result.all_seeds[idx] = processed.seed
            processed_result.infotexts[idx] = processed.infotexts[0]
        else:
            cell_mode = "P"
            cell_size = (processed_result.width, processed_result.height)
            if processed_result.images[0] is not None:
                cell_mode = processed_result.images[0].mode
                # This corrects size in case of batches:
                cell_size = processed_result.images[0].size
            processed_result.images[idx] = Image.new(cell_mode, cell_size)

    if first_axes_processed == "x":
        for ix, x in enumerate(xs):
            if second_axes_processed == "y":
                for iy, y in enumerate(ys):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == "y":
        for iy, y in enumerate(ys):
            if second_axes_processed == "x":
                for ix, x in enumerate(xs):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == "z":
        for iz, z in enumerate(zs):
            if second_axes_processed == "x":
                for ix, x in enumerate(xs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iy, y in enumerate(ys):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)

    if not processed_result:
        # Should never happen, I've only seen it on one of four open tabs and it needed to refresh.
        print("Unexpected error: Processing could not begin, you may need to refresh the tab or restart the service.")
        return Processed(p, [])
    elif not any(processed_result.images):
        print("Unexpected error: draw_xyz_grid failed to return even a single processed image")
        return Processed(p, [])

    z_count = len(zs)

    for i in range(z_count):
        start_index = (i * len(xs) * len(ys)) + i
        end_index = start_index + len(xs) * len(ys)
        grid = images.image_grid(processed_result.images[start_index:end_index], rows=len(ys))
        if draw_legend:
            grid = images.draw_grid_annotations(
                grid,
                processed_result.images[start_index].size[0],
                processed_result.images[start_index].size[1],
                hor_texts,
                ver_texts,
                margin_size,
            )
        processed_result.images.insert(i, grid)
        processed_result.all_prompts.insert(i, processed_result.all_prompts[start_index])
        processed_result.all_seeds.insert(i, processed_result.all_seeds[start_index])
        processed_result.infotexts.insert(i, processed_result.infotexts[start_index])

    sub_grid_size = processed_result.images[0].size
    z_grid = images.image_grid(processed_result.images[:z_count], rows=1)
    if draw_legend:
        z_grid = images.draw_grid_annotations(
            z_grid,
            sub_grid_size[0],
            sub_grid_size[1],
            title_texts,
            [[images.GridAnnotation()]],
        )
    processed_result.images.insert(0, z_grid)
    # TODO: Deeper aspects of the program rely on grid info being misaligned between metadata arrays, which is not ideal.
    # processed_result.all_prompts.insert(0, processed_result.all_prompts[0])
    # processed_result.all_seeds.insert(0, processed_result.all_seeds[0])
    processed_result.infotexts.insert(0, processed_result.infotexts[0])

    return processed_result


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*])?\s*")
