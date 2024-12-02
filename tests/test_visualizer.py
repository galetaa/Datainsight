import unittest
import pandas as pd
from visualizer import DataVisualizer


class TestTabularData(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Возраст': [23, 45, 31, 35, 26, 27, 30, 22, 40, 29, 33, 38],
            'Зарплата': [50000, 70000, 60000, 65000, 55000, 52000, 58000, 51000, 68000, 59000, 61000, 64000]
        })

    def test_histogram(self):
        visualizer = DataVisualizer(
            data=self.data,
            plot_type='histogram',
            x='Возраст',
            title='Распределение возраста',
        )
        fig = visualizer.create_plot()
        self.assertIsNotNone(fig)
        fig.show()

    def test_bar_chart(self):
        visualizer = DataVisualizer(
            data=self.data,
            plot_type='bar',
            x='Возраст',
            y='Зарплата',
            title='Зависимость зарплаты от возраста',
        )
        fig = visualizer.create_plot()
        self.assertIsNotNone(fig)
        fig.show()


class TestTextData(unittest.TestCase):
    def setUp(self):
        # Создаем текстовые данные
        self.text = "данные анализ визуализация python данные анализ данные визуализация данные данные обработка python анализ обработка данные"
        word_list = self.text.split()
        word_counts = pd.Series(word_list).value_counts().reset_index()
        word_counts.columns = ['Слово', 'Частота']
        self.data = word_counts

    def test_bar_chart_word_frequency(self):
        visualizer = DataVisualizer(
            data=self.data,
            plot_type='bar',
            x='Слово',
            y='Частота',
            title='Частота слов в тексте',
        )
        fig = visualizer.create_plot()
        self.assertIsNotNone(fig)
        fig.show()


class TestTimeSeriesData(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Дата': pd.date_range(start='2023-01-01', periods=12, freq='M'),
            'Значение': [100, 120, 130, 125, 150, 160, 170, 165, 180, 190, 200, 210]
        })

    def test_line_chart(self):
        visualizer = DataVisualizer(
            data=self.data,
            plot_type='line',
            x='Дата',
            y='Значение',
            title='Временной ряд значений',
        )
        fig = visualizer.create_plot()
        self.assertIsNotNone(fig)
        fig.show()


class TestImageData(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.data = np.random.rand(10, 10)

    def test_image_display(self):
        visualizer = DataVisualizer(
            data=self.data,
            plot_type='image',
            title='Пример изображения'
        )
        fig = visualizer.create_plot()
        self.assertIsNotNone(fig)
        fig.show()


class TestGeographicalData(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Город': ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Нижний Новгород', 'Казань',
                      'Челябинск', 'Омск', 'Самара', 'Ростов-на-Дону'],
            'Широта': [55.7558, 59.9343, 55.0084, 56.8389, 56.3269, 55.8304, 55.1644, 54.9885, 53.1959, 47.2357],
            'Долгота': [37.6176, 30.3351, 82.9357, 60.6057, 44.0075, 49.0661, 61.4368, 73.3845, 50.1008, 39.7015]
        })

    def test_scatter_geo(self):
        visualizer = DataVisualizer(
            data=self.data,
            plot_type='scatter_geo',
            lat='Широта',
            lon='Долгота',
            hover_name='Город',
            title='Города России',
            size_max=15
        )
        fig = visualizer.create_plot()
        self.assertIsNotNone(fig)
        fig.show()


class TestFinancialData(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Дата': pd.date_range(start='2023-01-01', periods=12, freq='D'),
            'Открытие': [100, 102, 101, 103, 105, 107, 106, 108, 110, 112, 111, 113],
            'Макс': [102, 104, 103, 105, 107, 109, 108, 110, 112, 114, 113, 115],
            'Мин': [99, 100, 99, 101, 103, 105, 104, 106, 108, 110, 109, 111],
            'Закрытие': [101, 103, 102, 104, 106, 108, 107, 109, 111, 113, 112, 114]
        })

    def test_candlestick(self):
        visualizer = DataVisualizer(
            data=self.data,
            plot_type='candlestick',
            x='Дата',
            open='Открытие',
            high='Макс',
            low='Мин',
            close='Закрытие',
            title='Свечной график акций'
        )
        fig = visualizer.create_plot()
        self.assertIsNotNone(fig)
        fig.show()


class TestUpdateDataMethod(unittest.TestCase):
    def setUp(self):
        self.data1 = pd.DataFrame({'X': range(10), 'Y': range(10, 20)})
        self.data2 = pd.DataFrame({'X': range(20, 30), 'Y': range(30, 40)})
        self.visualizer = DataVisualizer(
            data=self.data1,
            plot_type='scatter',
            x='X',
            y='Y',
            title='Первоначальные данные'
        )

    def test_update_data(self):
        self.visualizer.update_data(self.data2)
        fig = self.visualizer.create_plot()
        self.assertIsNotNone(fig)
        self.assertEqual(self.visualizer.data.equals(self.data2), True)


class TestUpdateKwargsMethod(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'X': range(10), 'Y': range(10, 20)})
        self.visualizer = DataVisualizer(
            data=self.data,
            plot_type='scatter',
            x='X',
            y='Y',
            title='Первоначальный заголовок'
        )

    def test_update_kwargs(self):
        self.visualizer.update_kwargs(title='Обновленный заголовок')
        fig = self.visualizer.create_plot()
        self.assertIsNotNone(fig)
        self.assertEqual(self.visualizer.kwargs['title'], 'Обновленный заголовок')


class TestSavePlotMethod(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'X': range(10), 'Y': range(10, 20)})
        self.visualizer = DataVisualizer(
            data=self.data,
            plot_type='scatter',
            x='X',
            y='Y',
            title='Тест сохранения графика'
        )

    def test_save_plot_html(self):
        self.visualizer.save_plot(file_path='test_plot.html', file_format='html')
        import os
        self.assertTrue(os.path.exists('test_plot.html'))
        os.remove('test_plot.html')

    def test_save_plot_png(self):
        self.visualizer.save_plot(file_path='test_plot.png', file_format='png')
        import os
        self.assertTrue(os.path.exists('test_plot.png'))
        os.remove('test_plot.png')
