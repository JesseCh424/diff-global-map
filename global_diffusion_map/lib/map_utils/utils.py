from shapely.geometry import LineString, box, Polygon, LinearRing
from shapely.geometry.base import BaseGeometry
from shapely import ops
import numpy as np
from scipy.spatial import distance
from typing import List, Optional, Tuple
from numpy.typing import NDArray


def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]:
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon',
                              'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"
    if 'Multi' in geom.geom_type:
        outs = []
        for g in geom.geoms:
            if g.is_valid and not g.is_empty:
                outs.append(g)
        return outs
    else:
        if geom.is_valid and not geom.is_empty:
            return [geom]
        else:
            return []


def get_drivable_area_contour(drivable_areas: List[Polygon],
                              roi_size: Tuple) -> List[LineString]:
    max_x = roi_size[0] / 2
    max_y = roi_size[1] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)

    exteriors = []
    interiors = []
    for poly in drivable_areas:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)

    results = []
    for ext in exteriors:
        if ext.is_ccw:
            ext = LinearRing(list(ext.coords)[::-1])
        lines = ext.intersection(local_patch)
        if lines.geom_type == 'GeometryCollection' and len(lines) == 0:
            continue
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        results.extend(split_collections(lines))

    for inter in interiors:
        if not inter.is_ccw:
            inter = LinearRing(list(inter.coords)[::-1])
        lines = inter.intersection(local_patch)
        if lines.geom_type == 'GeometryCollection' and len(lines) == 0:
            continue
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        results.extend(split_collections(lines))

    return results


def get_ped_crossing_contour(polygon: Polygon, local_patch: box) -> Optional[LineString]:
    ext = polygon.exterior
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    lines = ext.intersection(local_patch)
    if lines.type != 'LineString':
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)
        if lines.type != 'LineString':
            ls = []
            for l in lines.geoms:
                ls.append(np.array(l.coords))
            lines = np.concatenate(ls, axis=0)
            lines = LineString(lines)

    if not lines.is_empty:
        start = list(lines.coords[0])
        end = list(lines.coords[-1])
        if not np.allclose(start, end, atol=1e-3):
            new_line = list(lines.coords)
            new_line.append(start)
            lines = LineString(new_line)
        return lines
    return None


def remove_repeated_lines(lines: List[LineString]) -> List[LineString]:
    new_lines = []
    for line in lines:
        repeated = False
        for l in new_lines:
            length = min(line.length, l.length)
            area1 = line.buffer(0.1)
            area2 = l.buffer(0.1)
            inter = area1.intersection(area2).area
            union = area1.union(area2).area
            iou = inter / union
            if iou >= 0.90:
                repeated = True
                break
        if not repeated:
            new_lines.append(line)
    return new_lines


def remove_repeated_lanesegment(lane_dict):
    new_lane_dict = {}
    for key, value in lane_dict.items():
        repeated = False
        for _, new_value in new_lane_dict.items():
            line = LineString(value['polyline'].xyz)
            l = LineString(new_value['polyline'].xyz)
            area1 = line.buffer(0.01)
            area2 = l.buffer(0.01)
            inter = area1.intersection(area2).area
            union = area1.union(area2).area
            iou = inter / union
            if iou >= 0.90:
                repeated = True
                break
        if not repeated:
            new_lane_dict[key] = value
    return new_lane_dict


def reassign_graph_attribute(lane_dict):
    for key, value in lane_dict.items():
        if len(value['predecessors']) > 0:
            if value['predecessors'][0] not in lane_dict.keys() or value['predecessors'][0] == key:
                value['predecessors'] = []
            else:
                lane_dict[value['predecessors'][0]]['successors'] = [key]
    for key, value in lane_dict.items():
        if len(value['successors']) > 0:
            if value['successors'][0] not in lane_dict.keys() or value['successors'][0] == key:
                value['successors'] = []
            else:
                lane_dict[value['successors'][0]]['predecessors'] = [key]
    return lane_dict


def remove_boundary_dividers(dividers: List[LineString], boundaries: List[LineString]) -> List[LineString]:
    for idx in range(len(dividers))[::-1]:
        divider = dividers[idx]
        for bound in boundaries:
            length = min(divider.length, bound.length)
            if divider.buffer(0.3).intersection(bound.buffer(0.3)).area > 0.2 * length:
                dividers.pop(idx)
                break
    return dividers


def connect_lines(lines: List[LineString]) -> List[LineString]:
    new_lines = []
    eps = 0.1
    while len(lines) > 1:
        line1 = lines[0]
        merged_flag = False
        for i, line2 in enumerate(lines[1:]):
            begin1 = list(line1.coords)[0]
            end1 = list(line1.coords)[-1]
            begin2 = list(line2.coords)[0]
            end2 = list(line2.coords)[-1]
            dist_matrix = distance.cdist([begin1, end1], [begin2, end2])
            if dist_matrix[0, 0] < eps:
                coords = list(line2.coords)[::-1] + list(line1.coords)
            elif dist_matrix[0, 1] < eps:
                coords = list(line2.coords) + list(line1.coords)
            elif dist_matrix[1, 0] < eps:
                coords = list(line1.coords) + list(line2.coords)
            elif dist_matrix[1, 1] < eps:
                coords = list(line1.coords) + list(line2.coords)[::-1]
            else:
                continue
            new_line = LineString(coords)
            lines.pop(i + 1)
            lines[0] = new_line
            merged_flag = True
            break
        if merged_flag:
            continue
        new_lines.append(line1)
        lines.pop(0)
    if len(lines) == 1:
        new_lines.append(lines[0])
    return new_lines


def transform_from(xyz: NDArray, translation: NDArray, rotation: NDArray) -> NDArray:
    new_xyz = xyz @ rotation.T + translation
    return new_xyz

