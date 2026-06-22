from crowd.tiling import generate_tiles, merge_boxes


def test_generate_tiles_2x2_covers_frame():
    tiles = generate_tiles(1000, 800, grid="2x2", overlap=0.0)
    assert len(tiles) == 4
    # union covers corners
    assert any(t[0] == 0 and t[1] == 0 for t in tiles)
    assert any(t[2] == 1000 and t[3] == 800 for t in tiles)


def test_generate_tiles_overlap_widens_tiles():
    no = generate_tiles(1000, 800, grid="2x2", overlap=0.0)
    ov = generate_tiles(1000, 800, grid="2x2", overlap=0.2)
    # an overlapped interior tile is wider than the non-overlapped one
    assert (ov[0][2] - ov[0][0]) > (no[0][2] - no[0][0])


def test_merge_boxes_dedups_seam_duplicates():
    # same person detected in two tiles → near-identical boxes
    boxes = [(100, 100, 140, 200, 0.9), (102, 101, 141, 201, 0.8),
             (500, 300, 540, 400, 0.95)]
    merged = merge_boxes(boxes, iou_thresh=0.5)
    assert len(merged) == 2


def test_merge_boxes_keeps_distinct():
    boxes = [(0, 0, 40, 100, 0.9), (500, 500, 540, 600, 0.9)]
    assert len(merge_boxes(boxes, iou_thresh=0.5)) == 2
