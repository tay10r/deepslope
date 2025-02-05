"""
This script visualizes a height map in 3D using Panda3D.
The height map is read from a PNG file (named 'heightmap.png'),
and a terrain mesh is generated from it.
"""

from pathlib import Path

from loguru import logger

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    GeomVertexFormat, GeomVertexData, Geom, GeomNode, GeomTriangles,
    GeomVertexWriter, PNMImage, Vec3, Vec4, NodePath, AmbientLight,
    DirectionalLight, loadPrcFileData
)

from deepslope.config import Config, get_config

# Panda3D configuration settings
loadPrcFileData('', 'window-title Height Map Viewer')
loadPrcFileData('', 'win-size 800 600')


class HeightMapApp(ShowBase):
    def __init__(self, height_map_path: str, height_scale: float = 100.0) -> None:
        super().__init__()

        self.height_map: PNMImage = PNMImage()
        if not self.height_map.read(height_map_path):
            print(f'Failed to load {height_map_path}')
            exit(1)

        self.height_scale = height_scale
        self.terrain_node: NodePath | None = None

        self.generate_terrain()

        # Position the camera to get a decent view of the terrain.
        x_size: int = self.height_map.getXSize()
        y_size: int = self.height_map.getYSize()
        center: float = min(x_size, y_size) / 2.0
        self.cam.setPos(-center * 0.75, -center *
                        0.75, self.height_scale * 2.0)
        self.cam.lookAt(center, center, 0.0)

        self.setup_lighting()

    def generate_terrain(self) -> None:
        """
        Generates a terrain mesh from the height map.
        """

        x_size: int = self.height_map.getXSize()
        y_size: int = self.height_map.getYSize()

        # Define the vertex format (position, normal, and color)
        format: GeomVertexFormat = GeomVertexFormat.getV3n3c4()
        vdata: GeomVertexData = GeomVertexData(
            'terrain', format, GeomVertexData.UHStatic)

        # Create writers for vertex, normal, and color data
        vertex: GeomVertexWriter = GeomVertexWriter(vdata, 'vertex')
        normal: GeomVertexWriter = GeomVertexWriter(vdata, 'normal')
        color: GeomVertexWriter = GeomVertexWriter(vdata, 'color')

        # Create a 2D list to store vertices so that normals can be computed later.
        vertices: list[list[Vec3]] = [
            [Vec3(0, 0, 0) for _ in range(y_size)] for _ in range(x_size)]

        # Fill in the vertex data. Each pixel becomes a vertex.
        for x in range(x_size):
            for y in range(y_size):
                # Get the brightness (gray value) at the current pixel.
                brightness: float = self.height_map.getGray(x, y)
                # Calculate the vertex's height. (Adjust self.height_scale as needed.)
                height: float = brightness * self.height_scale
                vertex_pos: Vec3 = Vec3(float(x), float(y), height)
                vertices[x][y] = vertex_pos
                vertex.addData3f(vertex_pos)
                # Write a dummy normal (will update later)
                normal.addData3f(0, 0, 1)
                # Use the brightness for the vertex color (so the terrain is a nice grayscale).
                color.addData4f(brightness, brightness, brightness, 1.0)

        # Create a GeomTriangles primitive for the mesh.
        prim: GeomTriangles = GeomTriangles(Geom.UHStatic)

        # For each quad (two triangles), add vertices.
        for x in range(x_size - 1):
            for y in range(y_size - 1):
                # Vertex indices (row-major order): index = x * y_size + y
                v0: int = x * y_size + y
                v1: int = (x + 1) * y_size + y
                v2: int = (x + 1) * y_size + (y + 1)
                v3: int = x * y_size + (y + 1)

                prim.addVertices(v0, v1, v2)
                prim.addVertices(v0, v2, v3)

        # Compute normals for smooth lighting.
        num_vertices: int = x_size * y_size
        computed_normals: list[Vec3] = [
            Vec3(0, 0, 0) for _ in range(num_vertices)]

        # For each triangle, compute the face normal and accumulate it for each vertex.
        for i in range(prim.getNumPrimitives()):
            start: int = prim.getPrimitiveStart(i)
            end: int = prim.getPrimitiveEnd(i)
            for j in range(start, end, 3):
                idx0: int = prim.getVertex(j)
                idx1: int = prim.getVertex(j + 1)
                idx2: int = prim.getVertex(j + 2)
                # Convert flat index back to 2D grid coordinates.
                x0: int = idx0 // y_size
                y0: int = idx0 % y_size
                x1: int = idx1 // y_size
                y1: int = idx1 % y_size
                x2: int = idx2 // y_size
                y2: int = idx2 % y_size
                p0: Vec3 = vertices[x0][y0]
                p1: Vec3 = vertices[x1][y1]
                p2: Vec3 = vertices[x2][y2]
                # Compute the face normal via cross product.
                v_a: Vec3 = p1 - p0
                v_b: Vec3 = p2 - p0
                face_normal: Vec3 = v_a.cross(v_b)
                face_normal.normalize()
                computed_normals[idx0] += face_normal
                computed_normals[idx1] += face_normal
                computed_normals[idx2] += face_normal

        # Write the computed normals back to the vertex data.
        normal.setRow(0)
        for n in computed_normals:
            n.normalize()
            normal.setData3f(n)

        # Build the geometry and attach it to the scene graph.
        geom: Geom = Geom(vdata)
        geom.addPrimitive(prim)
        node: GeomNode = GeomNode('terrain')
        node.addGeom(geom)
        self.terrain_node = self.render.attachNewNode(node)
        self.terrain_node.setTwoSided(True)

    def setup_lighting(self) -> None:
        '''Sets up basic lighting for the scene.'''
        # Ambient light for general brightness.
        ambient: AmbientLight = AmbientLight('ambient')
        ambient.setColor(Vec4(0.5, 0.5, 0.5, 1))
        ambient_np: NodePath = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        # Directional light to create some nice shadows.
        directional: DirectionalLight = DirectionalLight('directional')
        directional.setColor(Vec4(0.7, 0.7, 0.7, 1))
        directional_np: NodePath = self.render.attachNewNode(directional)
        directional_np.setHpr(30.0, -60, 15)
        self.render.setLight(directional_np)


def main():
    config: Config = get_config()
    tmp_path = Path(config.tmp_path)
    if not tmp_path.exists():
        logger.error('The "tmp" directory is missing. Try training a model.')
        return

    test_images = list(tmp_path.glob('*.png'))
    if len(test_images) == 0:
        logger.error('There are no test images in the "tmp" directory.')
        return

    app: HeightMapApp = HeightMapApp(test_images[-1],
                                     height_scale=config.test_height)
    app.run()


if __name__ == '__main__':
    main()
