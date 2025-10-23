# %%
import numpy as np

t = np.linspace(0, 10, 5)  # 时间 [0, 2.5, 5, 7.5, 10]
x = t**2                    # 位移 [0, 6.25, 25, 56.25, 100]

dx_dt = np.gradient(x, t)   # 使用实际的时间间隔，更准确
# 或者 dx_dt = np.gradient(x, 2.5) # 使用固定的采样间隔 dt=2.5

print("时间 t: ", t)
print("位移 x: ", x)
print("速度 dx_dt: ", dx_dt) 
# 理论上 v = 2*t，所以在 t=5时，v=10。
# 输出会是一个接近 [5, 10, 15, 20, 25] 的数组，因为用的是中心差分。
# %%

# %%
import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的 2D 网格 (像是两个小山丘)
y, x = np.mgrid[-2:2:5j, -2:2:5j] # 5x5 的网格
print(x)
print(y)
z = np.exp(-(x**2 + y**2)) + 0.5 * np.exp(-((x-1)**2 + (y-1)**2))

# 计算梯度
dz_dy, dz_dx = np.gradient(z) 
# 注意：默认 axis=(0,1)，返回值顺序是 [grad_axis0, grad_axis1]
# 因为我们的网格是 (y, x)，所以 axis0 是 y 方向，axis1 是 x 方向。
# 所以 dz_dy 是沿 y 方向的变化率，dz_dx 是沿 x 方向的变化率。

print("地形高度 Z (5x5):")
print(z)
print("\nX方向梯度 (坡度) dz_dx (5x5):")
print(dz_dx) # 在东/西方向上的坡度
print("\nY方向梯度 (坡度) dz_dy (5x5):")
print(dz_dy) # 在北/南方向上的坡度

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.contourf(x, y, z, levels=20)
plt.colorbar(label='Height (Z)')
plt.title('Original Height Field')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 3, 2)
plt.contourf(x, y, dz_dx, levels=20)
plt.colorbar(label='dZ/dX')
plt.title('Gradient in X-direction (East/West Slope)')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 3, 3)
plt.contourf(x, y, dz_dy, levels=20)
plt.colorbar(label='dZ/dY')
plt.title('Gradient in Y-direction (North/South Slope)')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
# %%