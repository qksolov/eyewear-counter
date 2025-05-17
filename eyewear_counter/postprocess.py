import pandas as pd


def generate_report(predictions, errors_cnt, df_input=None, filename=None):
    """
    Генерирует итоговый отчёт на основе предсказаний модели и количества ошибок.

    Args:
        predictions (torch.Tensor or np.ndarray or list): Список предсказаний модели. 
            Ожидается массив shape=(N, 3), где каждый столбец соответствует 
            вероятности классов: ['В очках', 'Без очков', 'В солнцезащитных очках'].
        errors_cnt (int): Количество изображений, которые не удалось обработать.
        df_input (pd.DataFrame, optional): Входная таблица с метаинформацией.
            !!! Важно: порядок строк в df_input должен точно соответствовать порядку входных изображений, 
            для которых были получены predictions. Несоответствие может привести к некорректному отображению результатов.
        filename (str, optional): Если указан, результат сохраняется в Excel-файл с несколькими листами.

    Returns:
        tuple:
            - result_df (pd.DataFrame or None): Таблица с предсказаниями по строкам df_input.
            - summary_df (pd.DataFrame): Сводная таблица с общей статистикой.
            - class_isolation_stats_df (pd.DataFrame): Статистика по классам.
    """

    predictions_df = pd.DataFrame(predictions, columns=['В очках', 'Без очков', 'В солнцезащитных очках'])
    summary_df, class_isolation_stats_df = generate_summary(predictions_df, errors_cnt) 
    if df_input is None:
        result_df = None
    else:
        df_input = df_input.drop(columns=predictions_df.columns.intersection(df_input.columns), errors='ignore')
        result_df = pd.concat([df_input, predictions_df], axis=1)
        glasses_df = result_df[result_df['В очках'] > 0]
        plain_df = result_df[result_df['Без очков'] > 0]
        sunglasses_df = result_df[result_df['В солнцезащитных очках'] > 0]
        not_face_df = result_df[result_df[['В очках', 'Без очков', 'В солнцезащитных очках']].sum(axis=1) == 0]
    
        if filename:
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, sheet_name='Все результаты', index=False)
                summary_df.to_excel(writer, sheet_name='Сводная таблица', index=False)
                glasses_df.to_excel(writer, sheet_name='В очках', index=False)
                plain_df.to_excel(writer, sheet_name='Без очков', index=False)
                sunglasses_df.to_excel(writer, sheet_name='В солнцезащитных очках', index=False)
                not_face_df.to_excel(writer, sheet_name='Без лиц', index=False)
            
            print(f"Отчёт сохранён в файл: {filename}")
    return result_df, summary_df, class_isolation_stats_df



def generate_summary(df, errors_cnt):
    total_images = len(df)
    total_people = df[['В очках', 'Без очков', 'В солнцезащитных очках']].sum().sum()
    images_without_people = (df[['В очках', 'Без очков', 'В солнцезащитных очках']].sum(axis=1) == 0).sum()
    sums = df[['В очках', 'Без очков', 'В солнцезащитных очках']].sum()

    def count_images(only=None, has=None):
        if only:
            mask = (df[only] > 0) & df[[c for c in ['В очках', 'Без очков', 'В солнцезащитных очках'] if c != only]].eq(0).all(axis=1)
        else:
            mask = df[has] > 0
        return mask.sum()

    count_only_glasses = count_images(only='В очках')
    count_with_glasses = count_images(has='В очках')
    count_only_plain = count_images(only='Без очков')
    count_with_plain = count_images(has='Без очков')
    count_only_sunglasses = count_images(only='В солнцезащитных очках')
    count_with_sunglasses = count_images(has='В солнцезащитных очках')


    data_1 = [
        ('Обработано изображений', total_images - errors_cnt),
        ('Не загружено изображений', errors_cnt),
        ('Обнаружено людей', total_people),
    ]
    data_2 = [
        ('Всего людей в очках', sums['В очках']),
        ('Всего людей без очков', sums['Без очков']),
        ('Всего людей в солнцезащитных очках', sums['В солнцезащитных очках']),
    ]

    data_3 = [
        ('Изображений без людей', images_without_people - errors_cnt),
        ('Изображений, где все люди в очках', count_only_glasses),
        ('Изображений, где есть человек в очках', count_with_glasses),
        ('Изображений, где все люди без очков', count_only_plain),
        ('Изображений, где есть человек без очков', count_with_plain),
        ('Изображений, где все люди в солнцезащитных очках', count_only_sunglasses),
        ('Изображений, где есть человек в солнцезащитных очками', count_with_sunglasses),
    ]

    df_1 = pd.DataFrame(data_1, columns=['Описание', 'Количество'])
    df_2 = pd.DataFrame(data_2, columns=['Описание', 'Количество'])
    df_3 = pd.DataFrame(data_3, columns=['Описание', 'Количество'])

    df_1['Процент'] = ""
    if total_people > 0:
        df_2['Процент'] = (df_2['Количество'] / total_people * 100).apply(lambda x: f"{x:.1f}%")
    else:
        df_2['Процент'] = ""
    if total_images > 0:
        df_3['Процент'] = (df_3['Количество'] / total_images * 100).apply(lambda x: f"{x:.1f}%")
    else:
        df_3['Процент'] = ""

    separator = pd.DataFrame([["", "", ""]], columns=['Описание', 'Количество', 'Процент'])

    summary = pd.concat([df_1, separator, df_3, separator, df_2], ignore_index=True)

    class_isolation_stats = pd.DataFrame({
        "Категория": ["Очки", "Очки", "Без", "Без", "Солнце", "Солнце"],
        "Тип": ["Есть другие классы", "Без других классов"]*3,
        "Количество": [
            count_with_glasses - count_only_glasses, count_only_glasses,  
            count_with_plain - count_only_plain, count_only_plain, 
            count_with_sunglasses - count_only_sunglasses, count_only_sunglasses
        ]
    })
    return summary, class_isolation_stats
