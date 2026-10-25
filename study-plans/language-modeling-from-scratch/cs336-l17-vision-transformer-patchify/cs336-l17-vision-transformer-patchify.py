import torch

def vision_transformer_patchify(images, patch_height, patch_width):
    """
    Returns: dictionary containing patch tokens and grid coordinates
    """
    batch, channels, height, width = images.shape
    patch_rows = height // patch_height
    patch_columns = width // patch_width
    tokens = images.reshape(
        batch, channels, patch_rows, patch_height, patch_columns, patch_width
    ).permute(0, 2, 4, 1, 3, 5).reshape(
        batch,
        patch_rows * patch_columns,
        channels * patch_height * patch_width,
    )
    rows = torch.arange(patch_rows, device=images.device)
    columns = torch.arange(patch_columns, device=images.device)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    coordinates = torch.stack(
        (row_grid.reshape(-1), column_grid.reshape(-1)), dim=1
    )
    return {"tokens": tokens, "coordinates": coordinates}