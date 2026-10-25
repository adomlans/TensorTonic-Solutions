import torch

def multimodal_rope_coordinates(segments):
    """
    Returns: dictionary containing packed time-height-width coordinates
    """
    coordinates = []
    offset = 0
    for segment in segments:
        if segment["type"] == "text":
            for position in range(segment["length"]):
                value = offset + position
                coordinates.append((value, value, value))
            offset += segment["length"]
        elif segment["type"] == "separator":
            coordinates.append((offset, offset, offset))
            offset += 1
        elif segment["type"] == "image":
            height = segment["height"]
            width = segment["width"]
            for row in range(height):
                for column in range(width):
                    coordinates.append((offset, offset + row, offset + column))
            if height and width:
                offset += max(height, width)
        elif segment["type"] == "video":
            frames = segment["frames"]
            height = segment["height"]
            width = segment["width"]
            stride = segment.get("frame_stride", 1)
            for frame in range(frames):
                for row in range(height):
                    for column in range(width):
                        coordinates.append(
                            (offset + frame * stride, offset + row, offset + column)
                        )
            if frames and height and width:
                offset += max((frames - 1) * stride + 1, height, width)
    if not coordinates:
        tensor = torch.empty((3, 0), dtype=torch.int64)
    else:
        tensor = torch.tensor(coordinates, dtype=torch.int64).transpose(0, 1)
    return {"coordinates": tensor}
