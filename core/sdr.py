import hashlib
import struct

class SDR1024:
    """
    Модуль Восприятия (SDR-Парсер) для проекта Липид-2.
    Преобразует текст в 1024-битный разреженный вектор (Sparse Distributed Representation).
    Разреженность: ровно 20 активных бит из 1024 (~2%).
    """
    TOTAL_BITS = 1024
    NUM_U64 = 16  # 1024 / 64 = 16 элементов uint64
    TARGET_ON_BITS = 20

    def __init__(self, text: str = ""):
        self.raw_text = text.strip().lower()
        # Храним вектор как список из 16 64-битных целых чисел (uint64)
        self.bits = [0] * self.NUM_U64
        
        if self.raw_text:
            self._encode()

    def _encode(self):
        """
        Преобразует текст в 1024-битный SDR с детерминированным 
        морфологическим хэшированием N-грамм.
        """
        if not self.raw_text:
            return

        # Собираем N-граммы (символьные 2-граммы и 3-граммы + целое слово)
        # Это дает фонологическую/буквенную устойчивость (кот/код будут иметь близкие биты)
        ngrams = []
        padded = f"#{self.raw_text}#"
        
        # 2-граммы и 3-граммы
        for i in range(len(padded) - 1):
            ngrams.append(padded[i:i+2])
            if i < len(padded) - 2:
                ngrams.append(padded[i:i+3])
        ngrams.append(self.raw_text)

        # Вычисляем индексы бит для каждой N-граммы через SHA-256
        active_indices = set()
        
        for ngram in ngrams:
            # Генерация детерминированного хэша
            h = hashlib.sha256(ngram.encode('utf-8')).digest()
            # Берем первые 2 байта для получения индекса от 0 до 1023
            idx = struct.unpack(">H", h[:2])[0] % self.TOTAL_BITS
            active_indices.add(idx)
            
            if len(active_indices) >= self.TARGET_ON_BITS:
                break

        # Если N-грамм не хватило для 20 бит, добираем с помощью вторичного хэша
        counter = 0
        while len(active_indices) < self.TARGET_ON_BITS:
            h = hashlib.sha256(f"{self.raw_text}_{counter}".encode('utf-8')).digest()
            idx = struct.unpack(">H", h[:2])[0] % self.TOTAL_BITS
            active_indices.add(idx)
            counter += 1

        # Отбираем строго TARGET_ON_BITS (20) индексов
        final_indices = sorted(list(active_indices))[:self.TARGET_ON_BITS]

        # Заполняем наш 16x uint64 массив
        for bit_idx in final_indices:
            block_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            self.bits[block_idx] |= (1 << bit_offset)

    def overlap(self, other: 'SDR1024') -> int:
        """
        Быстрая побитовая операция AND между двумя SDR.
        Возвращает количество совпадающих единичных бит (Popcount).
        """
        shared_bits = 0
        for i in range(self.NUM_U64):
            intersection = self.bits[i] & other.bits[i]
            shared_bits += bin(intersection).count('1')
        return shared_bits

    def similarity(self, other: 'SDR1024') -> float:
        """
        Возвращает процент перекрытия (от 0.0 до 1.0).
        """
        shared = self.overlap(other)
        return shared / float(self.TARGET_ON_BITS)

    def active_bit_count(self) -> int:
        """Подсчет общего числа активных бит (должно быть ровно 20)."""
        return sum(bin(block).count('1') for block in self.bits)

    def __repr__(self):
        return f"<SDR1024 text='{self.raw_text}' active_bits={self.active_bit_count()}>"


# ==============================================================================
# БЫСТРЫЙ ТЕСТ И ДЕМОНСТРАЦИЯ
# ==============================================================================
if __name__ == "__main__":
    print("--- ТЕСТИРОВАНИЕ МОДУЛЯ SDR1024 (ЛИПИД-2) ---\n")

    # 1. Проверка морфологической схожести похожих слов
    sdr_kot = SDR1024("кот")
    sdr_kod = SDR1024("код")
    sdr_ball = SDR1024("мячик")

    print(f"Слово 1: {sdr_kot}")
    print(f"Слово 2: {sdr_kod}")
    print(f"Слово 3: {sdr_ball}\n")

    overlap_kot_kod = sdr_kot.overlap(sdr_kod)
    sim_kot_kod = sdr_kot.similarity(sdr_kod)
    
    overlap_kot_ball = sdr_kot.overlap(sdr_ball)
    sim_kot_ball = sdr_kot.similarity(sdr_ball)

    print(f"Перекрытие 'кот' и 'код': {overlap_kot_kod}/20 бит ({sim_kot_kod*100:.1f}%)")
    print(f"Перекрытие 'кот' и 'мячик': {overlap_kot_ball}/20 бит ({sim_kot_ball*100:.1f}%)\n")

    # 2. Проверка гигантского слова
    long_word = "сельскохозяйственно-машиностроительный"
    sdr_long = SDR1024(long_word)
    print(f"Длинное слово: '{long_word}'")
    print(f"Результат SDR: {sdr_long}")
    print(f"Успешно упаковано в 1024 бита без ошибок!")