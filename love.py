import pygame
import random
import math
import sys

# 初始化pygame
pygame.init()

# 设置屏幕尺寸
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("龙卷风动画")

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 龙卷风参数
num_particles = 500
particles = []

for _ in range(num_particles):
    angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(50, 200)
    speed = random.uniform(0.5, 2)
    color = [random.randint(0, 255) for _ in range(3)]
    particles.append({
        'angle': angle,
        'radius': radius,
        'speed': speed,
        'color': color,
        'y': HEIGHT
    })

# 火焰参数
flames = []

# 雷电参数
lightning = []
lightning_timer = 0

clock = pygame.time.Clock()

def draw_tornado():
    for particle in particles:
        # 更新位置
        particle['angle'] += 0.01 * particle['speed']
        particle['radius'] -= 0.5 * particle['speed']
        particle['y'] -= particle['speed']

        # 重新生成粒子
        if particle['radius'] < 20 or particle['y'] < HEIGHT / 2:
            particle['angle'] = random.uniform(0, 2 * math.pi)
            particle['radius'] = random.uniform(150, 300)
            particle['speed'] = random.uniform(0.5, 2)
            particle['color'] = [random.randint(0, 255) for _ in range(3)]
            particle['y'] = HEIGHT

        # 计算位置
        x = WIDTH / 2 + particle['radius'] * math.cos(particle['angle'])
        y = particle['y']

        # 绘制粒子
        pygame.draw.circle(screen, particle['color'], (int(x), int(y)), 2)

def draw_flames():
    # 添加新的火焰
    if random.randint(1, 10) == 1:
        flames.append({'x': WIDTH / 2, 'y': HEIGHT / 2, 'size': random.randint(5, 15), 'color': (255, random.randint(100, 200), 0)})

    # 更新和绘制火焰
    for flame in flames[:]:
        flame['y'] -= 2
        flame['size'] += 0.5
        flame['color'] = (flame['color'][0], max(flame['color'][1] - 2, 0), 0)
        pygame.draw.circle(screen, flame['color'], (int(flame['x']), int(flame['y'])), int(flame['size']))
        if flame['y'] < HEIGHT / 2 or flame['size'] > 30:
            flames.remove(flame)

def draw_lightning():
    global lightning_timer
    if lightning_timer <= 0:
        if random.randint(1, 100) == 1:
            start_x = random.randint(0, WIDTH)
            lightning.append({'points': [(start_x, 0)]})
            lightning_timer = random.randint(30, 60)
    else:
        lightning_timer -= 1

    for bolt in lightning[:]:
        last_point = bolt['points'][-1]
        if last_point[1] < HEIGHT / 2:
            new_x = last_point[0] + random.randint(-20, 20)
            new_y = last_point[1] + random.randint(10, 30)
            bolt['points'].append((new_x, new_y))
        else:
            # 绘制闪电
            pygame.draw.lines(screen, WHITE, False, bolt['points'], 2)
            lightning.remove(bolt)

def main():
    running = True
    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_tornado()
        draw_flames()
        draw_lightning()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
