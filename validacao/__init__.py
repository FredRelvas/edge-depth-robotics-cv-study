"""
Pipeline de validação offline (Frente 4).

Lê um rosbag gravado num run do TurtleBot4, projeta o LiDAR 2D sobre os pixels
da imagem, monta a tabela y_lidar / y_oak / y_ia e calcula as métricas de
profundidade comparando cada modelo com o baseline (RealSense) e o ground truth
(LiDAR). Ver o README.md na raiz do repositório.
"""
