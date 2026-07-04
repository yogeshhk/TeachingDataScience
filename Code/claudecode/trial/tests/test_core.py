from core import inspect_mesh, load_mesh


def test_thin_walls_flagged(thin_wall_mesh_path):
    mesh = load_mesh(thin_wall_mesh_path)
    report = inspect_mesh(mesh)

    assert "thin_walls" in report.issues
    assert not report.is_valid
