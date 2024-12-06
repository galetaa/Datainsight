import unittest
import numpy as np
from visualizer import Visualizer
import plotly.io as pio
import os


# -----------------------------------------
# Вспомогательные функции для генерации тестовых данных
# -----------------------------------------

def generate_linear_data(n=50):
    x = np.linspace(0, 10, n)
    y = 2 * x + 1
    return x, y


def generate_random_data(n=100):
    # Нормальное распределение
    return np.random.randn(n)


def generate_data_with_outliers(n=100, n_outliers=5):
    data = np.random.randn(n)
    data[:n_outliers] = data[:n_outliers] * 100  # искусственные выбросы
    return data


def generate_non_numeric_data(n=10):
    return ["a"] * n


def generate_nan_data(n=10):
    data = np.linspace(0, 1, n)
    data[3] = np.nan
    return data


def generate_inf_data(n=10):
    data = np.linspace(0, 1, n)
    data[-1] = np.inf
    return data


def generate_non_monotonic_x(n=10):
    x = np.array([3, 1, 2, 5, 4, 6, 10, 9, 8, 7])
    y = np.random.randn(n)
    return x, y


def generate_3d_scatter_data(n=30):
    x = np.linspace(0, 5, n)
    y = np.linspace(0, 5, n)
    z = x ** 2 + y
    return x, y, z


def generate_3d_surface_data(nx=30, ny=30):
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X ** 2 + Y ** 2)
    return Z


class TestVisualizerValidData(unittest.TestCase):
    """Тесты корректных данных для всех поддерживаемых типов визуализаций."""

    def test_histogram(self):
        v = Visualizer("histogram").load_data(x=generate_random_data(n=100))
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test histogram")
        self.assertIsNotNone(fig)

    def test_kde(self):
        v = Visualizer("kde").load_data(x=generate_random_data(n=200))
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test kde")
        self.assertIsNotNone(fig)

    def test_distribution(self):
        v = Visualizer("distribution").load_data(x=generate_random_data(n=50))
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test distribution")
        self.assertIsNotNone(fig)

    def test_scatter(self):
        x, y = generate_linear_data(n=50)
        v = Visualizer("scatter").load_data(x=x, y=y)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test scatter")
        self.assertIsNotNone(fig)

    def test_line(self):
        x, y = generate_linear_data(n=50)
        v = Visualizer("line").load_data(x=x, y=y)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test line")
        self.assertIsNotNone(fig)

    def test_bar(self):
        x, y = generate_linear_data(n=10)
        v = Visualizer("bar").load_data(x=x, y=y)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test bar")
        self.assertIsNotNone(fig)

    def test_area(self):
        x, y = generate_linear_data(n=50)
        v = Visualizer("area").load_data(x=x, y=y)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test area")
        self.assertIsNotNone(fig)

    def test_box(self):
        x, y = generate_linear_data(n=50)
        v = Visualizer("box").load_data(x=x, y=y)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test box")
        self.assertIsNotNone(fig)

    def test_violin(self):
        x, y = generate_linear_data(n=50)
        v = Visualizer("violin").load_data(x=x, y=y)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test violin")
        self.assertIsNotNone(fig)

    def test_scatter3d(self):
        x, y, z = generate_3d_scatter_data(n=30)
        v = Visualizer("scatter3d").load_data(x=x, y=y, z=z)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test scatter3d")
        self.assertIsNotNone(fig)

    def test_surface(self):
        z = generate_3d_surface_data(nx=20, ny=20)
        v = Visualizer("surface").load_data(z=z)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test surface")
        self.assertIsNotNone(fig)

    def test_contour(self):
        z = generate_3d_surface_data(nx=20, ny=20)
        v = Visualizer("contour").load_data(z=z)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test contour")
        self.assertIsNotNone(fig)

    def test_heatmap(self):
        z = generate_3d_surface_data(nx=20, ny=20)
        v = Visualizer("heatmap").load_data(z=z)
        self.assertTrue(v.validate())
        fig = v.get_figure(title="Test heatmap")
        self.assertIsNotNone(fig)


class TestVisualizerInvalidData(unittest.TestCase):
    """Тесты некорректных данных для различных типов визуализаций."""

    # 1D
    def test_histogram_few_points(self):
        v = Visualizer("histogram").load_data(x=[1])
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_1d_data_nan(self):
        v = Visualizer("histogram").load_data(x=generate_nan_data())
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_1d_data_inf(self):
        v = Visualizer("kde").load_data(x=generate_inf_data())
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_1d_data_outliers(self):
        v = Visualizer("distribution").load_data(x=generate_data_with_outliers(n=100, n_outliers=20))
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_1d_data_non_numeric(self):
        v = Visualizer("histogram").load_data(x=generate_non_numeric_data())
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    # 2D
    def test_2d_mismatch_length(self):
        v = Visualizer("scatter").load_data(x=[1, 2, 3], y=[1, 2])
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_2d_nan_values(self):
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        y[5] = np.nan
        v = Visualizer("line").load_data(x=x, y=y)
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_2d_inf_values(self):
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        y[-1] = np.inf
        v = Visualizer("bar").load_data(x=x, y=y)
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_2d_non_numeric(self):
        x = ["a", "b", "c"]
        y = [1, 2, 3]
        v = Visualizer("area").load_data(x=x, y=y)
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_2d_non_monotonic_line(self):
        x, y = generate_non_monotonic_x()
        v = Visualizer("line").load_data(x=x, y=y)
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    # 3D
    def test_3d_scatter_mismatch_length(self):
        x = [1, 2, 3]
        y = [4, 5, 6, 7]  # extra point
        z = [7, 8, 9]
        v = Visualizer("scatter3d").load_data(x=x, y=y, z=z)
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_3d_scatter_nan(self):
        x, y, z = generate_3d_scatter_data(n=10)
        z[3] = np.nan
        v = Visualizer("scatter3d").load_data(x=x, y=y, z=z)
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_3d_surface_not_2d(self):
        z = np.linspace(0, 10, 10)  # 1D массив вместо 2D
        v = Visualizer("surface").load_data(z=z)
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_3d_contour_inf(self):
        z = generate_3d_surface_data(10, 10)
        z[0, 0] = np.inf
        v = Visualizer("contour").load_data(z=z)
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    def test_3d_heatmap_non_numeric(self):
        z = np.array([["a", "b"], ["c", "d"]])
        v = Visualizer("heatmap").load_data(z=z)
        self.assertFalse(v.validate())
        with self.assertRaises(ValueError):
            v.plot()

    # Unsupported


class TestCombinedFiguresOnOnePage(unittest.TestCase):

    def test_all_figures_on_one_page(self):
        # Типы визуализаций, которые хотим отобразить
        visualization_types = [
            ("histogram", dict(x=np.random.randn(100))),
            ("kde", dict(x=np.random.randn(200))),
            ("distribution", dict(x=np.random.rand(50))),
            ("scatter", dict(x=np.linspace(0, 10, 50), y=np.linspace(0, 20, 50))),
            ("line", dict(x=np.linspace(0, 10, 50), y=np.sin(np.linspace(0, 10, 50)))),
            ("bar", dict(x=["A", "B", "C"], y=[10, 20, 15])),
            ("area", dict(x=np.linspace(0, 10, 50), y=np.cumsum(np.random.rand(50)))),
            ("box", dict(x=["group1"] * 25 + ["group2"] * 25,
                         y=np.concatenate((np.random.randn(25), np.random.randn(25) + 2)))),
            ("violin",
             dict(x=["cat1"] * 25 + ["cat2"] * 25, y=np.concatenate((np.random.randn(25), np.random.randn(25) + 3)))),
            ("scatter3d", dict(x=np.linspace(0, 5, 20), y=np.linspace(0, 5, 20), z=(np.linspace(0, 5, 20)) ** 2)),
            ("surface", dict(z=np.sin(np.linspace(0, 6, 30))[:, None] * np.linspace(0, 6, 30)[None, :])),
            ("contour", dict(z=(np.random.rand(20, 20) - 0.5) * 2)),
            ("heatmap", dict(z=np.random.rand(20, 20)))
        ]

        # Создаём HTML-код для всех фигур
        all_figs_html = "<html><head><title>All Figures</title></head><body>"
        all_figs_html += "<h1>All Visualizations</h1>"

        for vis_type, data_params in visualization_types:
            v = Visualizer(vis_type)
            v.load_data(**data_params)
            if v.validate():
                fig = v.get_figure(title=f"{vis_type.capitalize()} Chart")
                # Конвертируем фигуру в HTML
                fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
                # Добавим заголовок для каждого графика
                all_figs_html += f"<h2>{vis_type.capitalize()}</h2>"
                all_figs_html += fig_html
            else:
                # Если валидация не прошла, добавим сообщение
                all_figs_html += f"<h2>{vis_type.capitalize()} (Validation Failed)</h2>"

        all_figs_html += "</body></html>"

        # Сохраняем в файл
        output_filename = "all_visualizations.html"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(all_figs_html)

        # Проверим, что файл создался
        self.assertTrue(os.path.exists(output_filename), "HTML файл не был создан.")

        # Этот тест не проверяет корректность самих графиков,
        # но гарантирует, что мы смогли собрать все в одну страницу без ошибок.


if __name__ == '__main__':
    unittest.main()
