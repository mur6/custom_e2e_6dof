

        RT = np.zeros((4, 4))
        RT[3, 3] = 1
        RT[:3, :3] = objs_dict[idx]["R"]
        RT[:3, [3]] = objs_dict[idx]["T"]
        RT = np.linalg.inv(RT)
        RTs[idx] = RT[:3]
        center_homo = self.cam_intrinsic @ RT[:3, [3]]
        center = center_homo[:2] / center_homo[2]
        x = np.linspace(0, self.resolution[0] - 1, self.resolution[0])
        y = np.linspace(0, self.resolution[1] - 1, self.resolution[1])
        xv, yv = np.meshgrid(x, y)
        dx, dy = center[0] - xv, center[1] - yv
        distance = np.sqrt(dx**2 + dy**2)
        nx, ny = dx / distance, dy / distance
        Tz = np.ones((self.resolution[1], self.resolution[0])) * RT[2, 3]
        centermaps[idx] = np.array([nx, ny, Tz])
        centers[idx] = np.array([int(center[0]), int(center[1])])
        label[0] = 1 - label[1:].sum(axis=0)
        # Image.fromarray(label[0].astype(np.uint8) * 255).save("testlabel.png")
        # Image.open(rgb_path).save("testrgb.png")
        # cv2.imwrite("testcenter.png", img)
